from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def PromptForDeletion(refs, scope_name=None, prompt_title=None):
    """Prompts the user to confirm deletion of resources."""
    if not refs:
        return
    resource_type = CollectionToResourceType(refs[0].Collection())
    resource_name = CamelCaseToOutputFriendly(resource_type)
    prompt_list = []
    for ref in refs:
        if scope_name:
            ref_scope_name = scope_name
        elif hasattr(ref, 'region'):
            ref_scope_name = 'region'
        else:
            ref_scope_name = None
        if ref_scope_name:
            item = '[{0}] in [{1}]'.format(ref.Name(), getattr(ref, ref_scope_name))
        else:
            item = '[{0}]'.format(ref.Name())
        prompt_list.append(item)
    PromptForDeletionHelper(resource_name, prompt_list, prompt_title=prompt_title)