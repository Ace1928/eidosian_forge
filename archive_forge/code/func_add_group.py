from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
def add_group(self, help=None, category=None, mutex=False, required=False, hidden=False, sort_args=True, **kwargs):
    """Adds an argument group with mutex/required attributes to the parser.

    Args:
      help: str, The group help text description.
      category: str, The group flag category name, None for no category.
      mutex: bool, A mutually exclusive group if True.
      required: bool, A required group if True.
      hidden: bool, A hidden group if True.
      sort_args: bool, Whether to sort the group's arguments in help/usage text.
        NOTE - For ordering consistency across gcloud, generally prefer using
        argument categories to organize information (instead of unsetting the
        argument sorting).
      **kwargs: Passed verbatim to ArgumentInterceptor().

    Returns:
      The added argument object.
    """
    if 'description' in kwargs or 'title' in kwargs:
        raise parser_errors.ArgumentException('parser.add_group(): description or title kwargs not supported -- use help=... instead.')
    new_parser = super(type(self.parser), self.parser).add_argument_group()
    group = ArgumentInterceptor(parser=new_parser, is_global=self.is_global, cli_generator=self.cli_generator, allow_positional=self.allow_positional, data=self.data, help=help, category=category, mutex=mutex, required=required, hidden=hidden or self._is_hidden, sort_args=sort_args, **kwargs)
    self.arguments.append(group)
    return group