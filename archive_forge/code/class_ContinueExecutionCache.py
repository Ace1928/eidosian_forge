import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
class ContinueExecutionCache:
    cache = ExactWeakKeyDictionary()
    generated_code_metadata = ExactWeakKeyDictionary()

    @classmethod
    def lookup(cls, code, lineno, *key):
        if code not in cls.cache:
            cls.cache[code] = dict()
        key = tuple(key)
        if key not in cls.cache[code]:
            cls.cache[code][key] = cls.generate(code, lineno, *key)
        return cls.cache[code][key]

    @classmethod
    def generate(cls, code, lineno, offset: int, setup_fn_target_offsets: Tuple[int], nstack: int, argnames: Tuple[str], setup_fns: Tuple[ReenterWith], null_idxes: Tuple[int]) -> types.CodeType:
        assert offset is not None
        assert not code.co_flags & (CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR)
        assert code.co_flags & CO_OPTIMIZED
        if code in ContinueExecutionCache.generated_code_metadata:
            return cls.generate_based_on_original_code_object(code, lineno, offset, setup_fn_target_offsets, nstack, argnames, setup_fns, null_idxes)
        is_py311_plus = sys.version_info >= (3, 11)
        meta = ResumeFunctionMetadata(code)

        def update(instructions: List[Instruction], code_options: Dict[str, Any]):
            meta.instructions = copy.deepcopy(instructions)
            args = [f'___stack{i}' for i in range(nstack)]
            args.extend((v for v in argnames if v not in args))
            freevars = tuple(code_options['co_cellvars'] or []) + tuple(code_options['co_freevars'] or [])
            code_options['co_name'] = f'resume_in_{code_options['co_name']}'
            if is_py311_plus:
                code_options['co_qualname'] = f'resume_in_{code_options['co_qualname']}'
            code_options['co_firstlineno'] = lineno
            code_options['co_cellvars'] = tuple()
            code_options['co_freevars'] = freevars
            code_options['co_argcount'] = len(args)
            code_options['co_posonlyargcount'] = 0
            code_options['co_kwonlyargcount'] = 0
            code_options['co_varnames'] = tuple(args + [v for v in code_options['co_varnames'] if v not in args])
            code_options['co_flags'] = code_options['co_flags'] & ~(CO_VARARGS | CO_VARKEYWORDS)
            target = next((i for i in instructions if i.offset == offset))
            prefix = []
            if is_py311_plus:
                if freevars:
                    prefix.append(create_instruction('COPY_FREE_VARS', arg=len(freevars)))
                prefix.append(create_instruction('RESUME', arg=0))
            cleanup: List[Instruction] = []
            hooks = {fn.stack_index: fn for fn in setup_fns}
            hook_target_offsets = {fn.stack_index: setup_fn_target_offsets[i] for i, fn in enumerate(setup_fns)}
            offset_to_inst = {inst.offset: inst for inst in instructions}
            old_hook_target_remap = {}
            null_idxes_i = 0
            for i in range(nstack):
                while null_idxes_i < len(null_idxes) and null_idxes[null_idxes_i] == i + null_idxes_i:
                    prefix.append(create_instruction('PUSH_NULL'))
                    null_idxes_i += 1
                prefix.append(create_instruction('LOAD_FAST', argval=f'___stack{i}'))
                if i in hooks:
                    hook = hooks.pop(i)
                    hook_insts, exn_target = hook(code_options, cleanup)
                    prefix.extend(hook_insts)
                    if is_py311_plus:
                        hook_target_offset = hook_target_offsets.pop(i)
                        old_hook_target = offset_to_inst[hook_target_offset]
                        meta.prefix_block_target_offset_remap.append(hook_target_offset)
                        old_hook_target_remap[old_hook_target] = exn_target
            if is_py311_plus:
                meta.prefix_block_target_offset_remap = list(reversed(meta.prefix_block_target_offset_remap))
            assert not hooks
            prefix.append(create_jump_absolute(target))
            for inst in instructions:
                if inst.offset == target.offset:
                    break
                inst.starts_line = None
                if sys.version_info >= (3, 11):
                    inst.positions = None
            if cleanup:
                prefix.extend(cleanup)
                prefix.extend(cls.unreachable_codes(code_options))
            if old_hook_target_remap:
                assert is_py311_plus
                for inst in instructions:
                    if inst.exn_tab_entry and inst.exn_tab_entry.target in old_hook_target_remap:
                        inst.exn_tab_entry.target = old_hook_target_remap[inst.exn_tab_entry.target]
            instructions[:] = prefix + instructions
        new_code = transform_code_object(code, update)
        ContinueExecutionCache.generated_code_metadata[new_code] = meta
        return new_code

    @staticmethod
    def unreachable_codes(code_options) -> List[Instruction]:
        """Codegen a `raise None` to make analysis work for unreachable code"""
        return [create_instruction('LOAD_CONST', argval=None), create_instruction('RAISE_VARARGS', arg=1)]

    @classmethod
    def generate_based_on_original_code_object(cls, code, lineno, offset: int, setup_fn_target_offsets: Tuple[int, ...], *args):
        """
        This handles the case of generating a resume into code generated
        to resume something else.  We want to always generate starting
        from the original code object so that if control flow paths
        converge we only generated 1 resume function (rather than 2^n
        resume functions).
        """
        meta: ResumeFunctionMetadata = ContinueExecutionCache.generated_code_metadata[code]
        new_offset = None

        def find_new_offset(instructions: List[Instruction], code_options: Dict[str, Any]):
            nonlocal new_offset
            target, = (i for i in instructions if i.offset == offset)
            new_target, = (i2 for i1, i2 in zip(reversed(instructions), reversed(meta.instructions)) if i1 is target)
            assert target.opcode == new_target.opcode
            new_offset = new_target.offset
        transform_code_object(code, find_new_offset)
        if sys.version_info >= (3, 11):
            if not meta.block_target_offset_remap:
                block_target_offset_remap = meta.block_target_offset_remap = {}

                def remap_block_offsets(instructions: List[Instruction], code_options: Dict[str, Any]):
                    prefix_blocks: List[Instruction] = []
                    for inst in instructions:
                        if len(prefix_blocks) == len(meta.prefix_block_target_offset_remap):
                            break
                        if inst.opname == 'PUSH_EXC_INFO':
                            prefix_blocks.append(inst)
                    for inst, o in zip(prefix_blocks, meta.prefix_block_target_offset_remap):
                        block_target_offset_remap[cast(int, inst.offset)] = o
                    old_start_offset = cast(int, prefix_blocks[-1].offset) if prefix_blocks else -1
                    old_inst_offsets = sorted((n for n in setup_fn_target_offsets if n > old_start_offset))
                    targets = _filter_iter(instructions, old_inst_offsets, lambda inst, o: inst.offset == o)
                    new_targets = _filter_iter(zip(reversed(instructions), reversed(meta.instructions)), targets, lambda v1, v2: v1[0] is v2)
                    for new, old in zip(new_targets, targets):
                        block_target_offset_remap[old.offset] = new[1].offset
                transform_code_object(code, remap_block_offsets)
            setup_fn_target_offsets = tuple((block_target_offset_remap[n] for n in setup_fn_target_offsets))
        return ContinueExecutionCache.lookup(meta.code, lineno, new_offset, setup_fn_target_offsets, *args)