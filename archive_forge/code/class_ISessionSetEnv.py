from zope.interface import Attribute, Interface
class ISessionSetEnv(Interface):
    """A session that can set environment variables."""

    def setEnv(name, value):
        """
        Set an environment variable for the shell or command to be started.

        From U{RFC 4254, section 6.4
        <https://tools.ietf.org/html/rfc4254#section-6.4>}: "Uncontrolled
        setting of environment variables in a privileged process can be a
        security hazard.  It is recommended that implementations either
        maintain a list of allowable variable names or only set environment
        variables after the server process has dropped sufficient
        privileges."

        (OpenSSH refuses all environment variables by default, but has an
        C{AcceptEnv} configuration option to select specific variables to
        accept.)

        @param name: The name of the environment variable to set.
        @type name: L{bytes}
        @param value: The value of the environment variable to set.
        @type value: L{bytes}
        @raise EnvironmentVariableNotPermitted: if setting this environment
            variable is not permitted.
        """