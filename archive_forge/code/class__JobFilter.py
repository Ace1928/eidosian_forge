from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import job_display
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import times
class _JobFilter(object):
    """Predicate for filtering jobs."""

    def __init__(self, args):
        """Create a _JobFilter from the given args.

    Args:
      args: The argparse.Namespace containing the parsed arguments.
    """
        self.preds = []
        if args.created_after or args.created_before:
            self._ParseTimePredicate(args.created_after, args.created_before)

    def __call__(self, job):
        return all([pred(job) for pred in self.preds])

    def _ParseTimePredicate(self, after, before):
        """Return a predicate for filtering jobs by their creation time.

    Args:
      after: Only return true if the job was created after this time.
      before: Only return true if the job was created before this time.
    """
        if after and (not before):
            self.preds.append(lambda x: times.ParseDateTime(x.createTime) > after)
        elif not after and before:
            self.preds.append(lambda x: times.ParseDateTime(x.createTime) <= before)
        elif after and before:

            def _Predicate(x):
                create_time = times.ParseDateTime(x.createTime)
                return after < create_time and create_time <= before
            self.preds.append(_Predicate)