from __future__ import annotations
import argparse
import enum
import functools
import typing as t
from ..constants import (
from ..util import (
from ..completion import (
from ..cli.argparsing import (
from ..cli.argparsing.actions import (
from ..cli.actions import (
from ..cli.compat import (
from ..config import (
from .completers import (
from .converters import (
from .epilog import (
from ..ci import (
def add_environments(parser: argparse.ArgumentParser, completer: CompositeActionCompletionFinder, controller_mode: ControllerMode, target_mode: TargetMode) -> None:
    """Add arguments for the environments used to run ansible-test and commands it invokes."""
    no_environment = controller_mode == ControllerMode.NO_DELEGATION and target_mode == TargetMode.NO_TARGETS
    parser.set_defaults(no_environment=no_environment)
    if no_environment:
        return
    parser.set_defaults(target_mode=target_mode)
    add_global_options(parser, controller_mode)
    add_legacy_environment_options(parser, controller_mode, target_mode)
    action_types = add_composite_environment_options(parser, completer, controller_mode, target_mode)
    sections = [f'{heading}\n{content}' for action_type, documentation_state in CompositeAction.documentation_state.items() if action_type in action_types for heading, content in documentation_state.sections.items()]
    if not get_ci_provider().supports_core_ci_auth():
        sections.append('Remote provisioning options have been hidden since no Ansible Core CI API key was found.')
    sections.append(get_epilog(completer))
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = '\n\n'.join(sections)