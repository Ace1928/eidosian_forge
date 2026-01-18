from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class Terminate(base.Command):
    """Terminate an active session."""
    detailed_help = {'EXAMPLES': '          To terminate a session "my-session" in the "us-central1" location, run:\n\n            $ {command} my-session --location=us-central1\n          '}

    @staticmethod
    def Args(parser):
        base.ASYNC_FLAG.AddToParser(parser)
        flags.AddTimeoutFlag(parser)
        dataproc = dp.Dataproc()
        flags.AddSessionResourceArg(parser, 'terminate', dataproc.api_version)

    def Run(self, args):
        dataproc = dp.Dataproc()
        session_id = args.CONCEPTS.session.Parse()
        console_io.PromptContinue(message="The session '{0}' will be terminated.".format(session_id.Name()), cancel_on_no=True, cancel_string='Termination canceled by user.')
        request = dataproc.messages.DataprocProjectsLocationsSessionsTerminateRequest(name=session_id.RelativeName())
        operation = dataproc.client.projects_locations_sessions.Terminate(request)
        if args.async_:
            log.status.write("Terminating session '{0}'".format(session_id.Name()))
            return operation
        operation = util.WaitForOperation(dataproc, operation, message="Waiting for session '{0}' to terminate.".format(session_id.Name()), timeout_s=args.timeout)
        log.DeletedResource(session_id.RelativeName())
        return operation