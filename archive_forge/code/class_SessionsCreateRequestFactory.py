from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc.sessions import session_message_factory
class SessionsCreateRequestFactory(object):
    """Factory class handling SessionsCreateRequest message.

  Factory class for configure argument parser and create
  SessionsCreateRequest message from parsed argument.
  """

    def __init__(self, dataproc, session_message_factory_override=None):
        """Factory for SessionsCreateRequest message.

    Only handles general create flags added by this class. User needs to
    provide session specific message when creating the request message.

    Args:
      dataproc: A api_lib.dataproc.Dataproc instance.
      session_message_factory_override: Override SessionMessageFactory instance.
    """
        self.dataproc = dataproc
        self.session_message_factory = session_message_factory_override
        if not self.session_message_factory:
            self.session_message_factory = session_message_factory.SessionMessageFactory(self.dataproc)

    def GetRequest(self, args):
        """Creates a SessionsCreateRequest message.

    Creates a SessionsCreateRequest message. The factory only handles the
    arguments added in AddArguments function. User needs to provide session
    specific message instance.

    Args:
      args: Parsed arguments.

    Returns:
      SessionsCreateRequest: A configured SessionsCreateRequest.
    """
        kwargs = {}
        kwargs['parent'] = args.CONCEPTS.session.Parse().Parent().RelativeName()
        kwargs['requestId'] = args.request_id
        if not kwargs['requestId']:
            kwargs['requestId'] = util.GetUniqueId()
        kwargs['sessionId'] = args.session
        kwargs['session'] = self.session_message_factory.GetMessage(args)
        return self.dataproc.messages.DataprocProjectsLocationsSessionsCreateRequest(**kwargs)