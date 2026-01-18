import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
class BaseInstr(Generic[A]):
    """Abstract instruction."""
    __slots__ = ('_name', '_opcode', '_arg', '_location')

    def __init__(self, name: str, arg: A=UNSET, *, lineno: Union[int, None, _UNSET]=UNSET, location: Optional[InstrLocation]=None) -> None:
        self._set(name, arg)
        if location:
            self._location = location
        elif lineno is UNSET:
            self._location = None
        else:
            self._location = InstrLocation(lineno, None, None, None)

    def set(self, name: str, arg: A=UNSET) -> None:
        """Modify the instruction in-place.

        Replace name and arg attributes. Don't modify lineno.

        """
        self._set(name, arg)

    def require_arg(self) -> bool:
        """Does the instruction require an argument?"""
        return opcode_has_argument(self._opcode)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._set(name, self._arg)

    @property
    def opcode(self) -> int:
        return self._opcode

    @opcode.setter
    def opcode(self, op: int) -> None:
        if not isinstance(op, int):
            raise TypeError('operator code must be an int')
        if 0 <= op <= 255:
            name = _opcode.opname[op]
            valid = name != '<%r>' % op
        else:
            valid = False
        if not valid:
            raise ValueError('invalid operator code')
        self._set(name, self._arg)

    @property
    def arg(self) -> A:
        return self._arg

    @arg.setter
    def arg(self, arg: A):
        self._set(self._name, arg)

    @property
    def lineno(self) -> Union[int, _UNSET, None]:
        return self._location.lineno if self._location is not None else UNSET

    @lineno.setter
    def lineno(self, lineno: Union[int, _UNSET, None]) -> None:
        loc = self._location
        if loc and (loc.end_lineno is not None or loc.col_offset is not None or loc.end_col_offset is not None):
            raise RuntimeError('The lineno of an instruction with detailed location information cannot be set.')
        if lineno is UNSET:
            self._location = None
        else:
            self._location = InstrLocation(lineno, None, None, None)

    @property
    def location(self) -> Optional[InstrLocation]:
        return self._location

    @location.setter
    def location(self, location: Optional[InstrLocation]) -> None:
        if location and (not isinstance(location, InstrLocation)):
            raise TypeError('The instr location must be an instance of InstrLocation or None.')
        self._location = location

    def stack_effect(self, jump: Optional[bool]=None) -> int:
        if not self.require_arg():
            arg = None
        elif self.name in BITFLAG_INSTRUCTIONS and isinstance(self._arg, tuple):
            assert len(self._arg) == 2
            arg = self._arg[0]
        elif self.name in BITFLAG2_INSTRUCTIONS and isinstance(self._arg, tuple):
            assert len(self._arg) == 3
            arg = self._arg[0]
        elif not isinstance(self._arg, int) or self._opcode in _opcode.hasconst:
            arg = 0
        else:
            arg = self._arg
        return dis.stack_effect(self._opcode, arg, jump=jump)

    def pre_and_post_stack_effect(self, jump: Optional[bool]=None) -> Tuple[int, int]:
        _effect = self.stack_effect(jump=jump)
        n = self.name
        if n in STATIC_STACK_EFFECTS:
            return STATIC_STACK_EFFECTS[n]
        elif n in DYNAMIC_STACK_EFFECTS:
            return DYNAMIC_STACK_EFFECTS[n](_effect, self.arg, jump)
        else:
            return (_effect, 0)

    def copy(self: T) -> T:
        return self.__class__(self._name, self._arg, location=self._location)

    def has_jump(self) -> bool:
        return self._has_jump(self._opcode)

    def is_cond_jump(self) -> bool:
        """Is a conditional jump?"""
        name = self._name
        return 'JUMP_' in name and 'IF_' in name

    def is_uncond_jump(self) -> bool:
        """Is an unconditional jump?"""
        return self.name in {'JUMP_FORWARD', 'JUMP_ABSOLUTE', 'JUMP_BACKWARD', 'JUMP_BACKWARD_NO_INTERRUPT'}

    def is_abs_jump(self) -> bool:
        """Is an absolute jump."""
        return self._opcode in _opcode.hasjabs

    def is_forward_rel_jump(self) -> bool:
        """Is a forward relative jump."""
        return self._opcode in _opcode.hasjrel and 'BACKWARD' not in self._name

    def is_backward_rel_jump(self) -> bool:
        """Is a backward relative jump."""
        return self._opcode in _opcode.hasjrel and 'BACKWARD' in self._name

    def is_final(self) -> bool:
        if self._name in {'RETURN_VALUE', 'RETURN_CONST', 'RAISE_VARARGS', 'RERAISE', 'BREAK_LOOP', 'CONTINUE_LOOP'}:
            return True
        if self.is_uncond_jump():
            return True
        return False

    def __repr__(self) -> str:
        if self._arg is not UNSET:
            return '<%s arg=%r location=%s>' % (self._name, self._arg, self._location)
        else:
            return '<%s location=%s>' % (self._name, self._location)

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self._cmp_key() == other._cmp_key()
    _name: str
    _location: Optional[InstrLocation]
    _opcode: int
    _arg: A

    def _set(self, name: str, arg: A) -> None:
        if not isinstance(name, str):
            raise TypeError('operation name must be a str')
        try:
            opcode = _opcode.opmap[name]
        except KeyError:
            raise ValueError(f'invalid operation name: {name}')
        if opcode >= MIN_INSTRUMENTED_OPCODE:
            raise ValueError(f'operation {name} is an instrumented or pseudo opcode. Only base opcodes are supported')
        self._check_arg(name, opcode, arg)
        self._name = name
        self._opcode = opcode
        self._arg = arg

    @staticmethod
    def _has_jump(opcode) -> bool:
        return opcode in _opcode.hasjrel or opcode in _opcode.hasjabs

    @abstractmethod
    def _check_arg(self, name: str, opcode: int, arg: A) -> None:
        pass

    @abstractmethod
    def _cmp_key(self) -> Tuple[Optional[InstrLocation], str, Any]:
        pass