from __future__ import annotations
import glob
import os
import re
import pathlib
import shutil
import subprocess
import typing as T
import functools
from mesonbuild.interpreterbase.decorators import FeatureDeprecated
from .. import mesonlib, mlog
from ..environment import get_llvm_tool_names
from ..mesonlib import version_compare, version_compare_many, search_version, stringlistify, extract_as_list
from .base import DependencyException, DependencyMethods, detect_compiler, strip_system_includedirs, strip_system_libdirs, SystemDependency, ExternalDependency, DependencyTypeName
from .cmake import CMakeDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .misc import threads_factory
from .pkgconfig import PkgConfigDependency
class JDKSystemDependency(JNISystemDependency):

    def __init__(self, environment: 'Environment', kwargs: JNISystemDependencyKW):
        super().__init__(environment, kwargs)
        self.feature_since = ('0.59.0', '')
        self.featurechecks.append(FeatureDeprecated('jdk system dependency', '0.62.0', 'Use the jni system dependency instead'))