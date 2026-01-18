from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.api_lib.dataproc.poller import session_poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc.sessions import (
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class Spark(base.Command):
    """Create a Spark session."""
    detailed_help = {'DESCRIPTION': '          Create a Spark session.\n          ', 'EXAMPLES': '          To create a Spark session, to:\n\n            $ {command} my-session --location=us-central1\n          '}

    @staticmethod
    def Args(parser):
        flags.AddSessionResourceArg(parser, 'create', dp.Dataproc().api_version)

    def Run(self, args):
        dataproc = dp.Dataproc()
        request = sessions_create_request_factory.SessionsCreateRequestFactory(dataproc).GetRequest(args)
        session_op = dataproc.client.projects_locations_sessions.Create(request)
        log.status.Print('Waiting for session creation operation...')
        metadata = util.ParseOperationJsonMetadata(session_op.metadata, dataproc.messages.SessionOperationMetadata)
        for warning in metadata.warnings:
            log.warning(warning)
        if not args.async_:
            poller = session_poller.SessionPoller(dataproc)
            waiter.WaitFor(poller, '{}/sessions/{}'.format(request.parent, request.sessionId), max_wait_ms=sys.maxsize, sleep_ms=5000, wait_ceiling_ms=5000, exponential_sleep_multiplier=1.0, custom_tracker=None, tracker_update_func=poller.TrackerUpdateFunction)
            log.status.Print('Session [{}] is created.'.format(request.sessionId))
        return session_op