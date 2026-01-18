from __future__ import annotations
import pathlib
import typing as T
from mesonbuild import mesonlib
from mesonbuild.build import CustomTarget, CustomTargetIndex, GeneratedList, Target
from mesonbuild.compilers import detect_compiler_for
from mesonbuild.interpreterbase.decorators import ContainerTypeInfo, FeatureDeprecated, FeatureNew, KwargInfo, typed_pos_args, typed_kwargs
from mesonbuild.mesonlib import version_compare, MachineChoice
from . import NewExtensionModule, ModuleReturnValue, ModuleInfo
from ..interpreter.type_checking import NoneType
def __native_headers(self, state: ModuleState, args: T.Tuple[T.List[mesonlib.FileOrString]], kwargs: T.Dict[str, T.Optional[str]]) -> ModuleReturnValue:
    classes = T.cast('T.List[str]', kwargs.get('classes'))
    package = kwargs.get('package')
    if package:
        sanitized_package = package.replace('-', '_').replace('.', '_')
    headers: T.List[str] = []
    for clazz in classes:
        sanitized_clazz = clazz.replace('.', '_')
        if package:
            headers.append(f'{sanitized_package}_{sanitized_clazz}.h')
        else:
            headers.append(f'{sanitized_clazz}.h')
    javac = self.__get_java_compiler(state)
    command = mesonlib.listify([javac.exelist, '-d', '@PRIVATE_DIR@', '-h', state.subdir, '@INPUT@'])
    prefix = classes[0] if not package else package
    target = CustomTarget(f'{prefix}-native-headers', state.subdir, state.subproject, state.environment, command, sources=args[0], outputs=headers, backend=state.backend)
    if version_compare(javac.version, '1.8.0'):
        pathlib.Path(state.backend.get_target_private_dir_abs(target)).mkdir(parents=True, exist_ok=True)
    return ModuleReturnValue(target, [target])