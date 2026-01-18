from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def GetCommandMetaData(command):
    from googlecloudsdk.core.document_renderers import render_document
    command_metadata = render_document.CommandMetaData()
    for arg in command.GetAllAvailableFlags():
        for arg_name in arg.option_strings:
            command_metadata.flags.append(arg_name)
            if isinstance(arg, argparse._StoreConstAction):
                command_metadata.bool_flags.append(arg_name)
    command_metadata.is_group = command.is_group
    return command_metadata