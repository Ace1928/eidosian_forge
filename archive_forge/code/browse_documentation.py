from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import browser_dispatcher
from googlecloudsdk.command_lib.app import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Launch a browser, or return a table of URLs to print.