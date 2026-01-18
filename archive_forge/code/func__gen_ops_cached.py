import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional
import sympy
import torch
from ...codecache import cache_dir
from ...config import cuda as inductor_cuda_config
from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version
@functools.lru_cache(None)
def _gen_ops_cached(arch, version) -> List[Any]:
    assert try_import_cutlass()
    import cutlass_library.generator as cutlass_generator
    import cutlass_library.manifest as cutlass_manifest
    if arch is None or version is None:
        log.error('Cannot detect cuda arch %s or cuda version %s. Will discard all cutlass ops. Please consider setting _inductor.cuda.arch and _inductor.cuda.version configs.', arch, version)
        return list()
    arch = _normalize_cuda_arch(arch)
    args = CUTLASSArgs(architectures=arch, cuda_version=version)
    manifest = cutlass_manifest.Manifest(args)
    if arch == '90':
        cutlass_generator.GenerateSM90(manifest, args.cuda_version)
        cutlass_generator.GenerateSM80(manifest, args.cuda_version)
    else:
        try:
            func = getattr(cutlass_generator, 'GenerateSM' + arch)
            func(manifest, args.cuda_version)
        except AttributeError as e:
            raise NotImplementedError('Arch ' + arch + ' is not supported by current cutlass lib.') from e
    return manifest.operations