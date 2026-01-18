from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer
from twisted.persisted import sob
from twisted.plugin import IPlugin
from twisted.python import components
from twisted.python.reflect import namedAny
class IServiceMaker(Interface):
    """
    An object which can be used to construct services in a flexible
    way.

    This interface should most often be implemented along with
    L{twisted.plugin.IPlugin}, and will most often be used by the
    'twistd' command.
    """
    tapname = Attribute("A short string naming this Twisted plugin, for example 'web' or 'pencil'. This name will be used as the subcommand of 'twistd'.")
    description = Attribute('A brief summary of the features provided by this Twisted application plugin.')
    options = Attribute('A C{twisted.python.usage.Options} subclass defining the configuration options for this application.')

    def makeService(options):
        """
        Create and return an object providing
        L{twisted.application.service.IService}.

        @param options: A mapping (typically a C{dict} or
        L{twisted.python.usage.Options} instance) of configuration
        options to desired configuration values.
        """