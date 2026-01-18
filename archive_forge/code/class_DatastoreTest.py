import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import datastores
class DatastoreTest(testtools.TestCase):

    def setUp(self):
        super(DatastoreTest, self).setUp()
        self.orig__init = datastores.Datastore.__init__
        datastores.Datastore.__init__ = mock.Mock(return_value=None)
        self.datastore = datastores.Datastore()
        self.datastore.manager = mock.Mock()

    def tearDown(self):
        super(DatastoreTest, self).tearDown()
        datastores.Datastore.__init__ = self.orig__init

    def test___repr__(self):
        self.datastore.name = 'datastore-1'
        self.assertEqual('<Datastore: datastore-1>', self.datastore.__repr__())