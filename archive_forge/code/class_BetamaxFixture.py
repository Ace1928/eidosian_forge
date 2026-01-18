from functools import partial
from unittest import mock
import betamax
import fixtures
import requests
from keystoneauth1.fixture import hooks
from keystoneauth1.fixture import serializer as yaml_serializer
from keystoneauth1 import session
class BetamaxFixture(fixtures.Fixture):

    def __init__(self, cassette_name, cassette_library_dir=None, serializer=None, record=False, pre_record_hook=hooks.pre_record_hook, serializer_name=None, request_matchers=None):
        """Configure Betamax for the test suite.

        :param str cassette_name:
            This is simply the name of the cassette without any file extension
            or containing directory. For example, to generate
            ``keystoneauth1/tests/unit/data/example.yaml``, one would pass
            only ``example``.
        :param str cassette_library_dir:
            This is the directory that will contain all cassette files. In
            ``keystoneauth1/tests/unit/data/example.yaml`` you would pass
            ``keystoneauth1/tests/unit/data/``.
        :param serializer:
            A class that implements the Serializer API in Betamax. See also:
            https://betamax.readthedocs.io/en/latest/serializers.html
        :param record:
            The Betamax record mode to use. If ``False`` (the default), then
            Betamax will not record anything. For more information about
            record modes, see:
            https://betamax.readthedocs.io/en/latest/record_modes.html
        :param callable pre_record_hook:
            Function or callable to use to perform some handling of the
            request or response data prior to saving it to disk.
        :param str serializer_name:
            The name of a serializer already registered with Betamax to use
            to handle cassettes. For example, if you want to use the default
            Betamax serializer, you would pass ``'json'`` to this parameter.
        :param list request_matchers:
            The list of request matcher names to use with Betamax. Betamax's
            default list is used if none are specified. See also:
            https://betamax.readthedocs.io/en/latest/matchers.html
        """
        self.cassette_library_dir = cassette_library_dir
        self.record = record
        self.cassette_name = cassette_name
        if not (serializer or serializer_name):
            serializer = yaml_serializer.YamlJsonSerializer
            serializer_name = serializer.name
        if serializer:
            betamax.Betamax.register_serializer(serializer)
        self.serializer = serializer
        self._serializer_name = serializer_name
        self.pre_record_hook = pre_record_hook
        self.use_cassette_kwargs = {}
        if request_matchers is not None:
            self.use_cassette_kwargs['match_requests_on'] = request_matchers

    @property
    def serializer_name(self):
        """Determine the name of the selected serializer.

        If a class was specified, use the name attribute to generate this,
        otherwise, use the serializer_name parameter from ``__init__``.

        :returns:
            Name of the serializer
        :rtype:
            str
        """
        if self.serializer:
            return self.serializer.name
        return self._serializer_name

    def setUp(self):
        super(BetamaxFixture, self).setUp()
        self.mockpatch = mock.patch.object(session, '_construct_session', partial(_construct_session_with_betamax, self))
        self.mockpatch.start()
        self.addCleanup(self.mockpatch.stop)