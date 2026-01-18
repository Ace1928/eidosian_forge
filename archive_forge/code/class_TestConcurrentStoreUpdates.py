import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
class TestConcurrentStoreUpdates(TestStore):
    """Test that Stores properly handle conccurent updates.

    New Store implementation may fail some of these tests but until such
    implementations exist it's hard to properly filter them from the scenarios
    applied here. If you encounter such a case, contact the bzr devs.
    """
    scenarios = [(key, {'get_stack': builder}) for key, builder in config.test_stack_builder_registry.iteritems()]

    def setUp(self):
        super().setUp()
        self.stack = self.get_stack(self)
        if not isinstance(self.stack, config._CompatibleStack):
            raise tests.TestNotApplicable('%s is not meant to be compatible with the old config design' % (self.stack,))
        self.stack.set('one', '1')
        self.stack.set('two', '2')
        self.stack.store.save()

    def test_simple_read_access(self):
        self.assertEqual('1', self.stack.get('one'))

    def test_simple_write_access(self):
        self.stack.set('one', 'one')
        self.assertEqual('one', self.stack.get('one'))

    def test_listen_to_the_last_speaker(self):
        c1 = self.stack
        c2 = self.get_stack(self)
        c1.set('one', 'ONE')
        c2.set('two', 'TWO')
        self.assertEqual('ONE', c1.get('one'))
        self.assertEqual('TWO', c2.get('two'))
        self.assertEqual('ONE', c2.get('one'))

    def test_last_speaker_wins(self):
        c1 = self.stack
        c2 = self.get_stack(self)
        c1.set('one', 'c1')
        c2.set('one', 'c2')
        self.assertEqual('c2', c2.get('one'))
        self.assertEqual('c1', c1.get('one'))
        c1.set('two', 'done')
        self.assertEqual('c2', c1.get('one'))

    def test_writes_are_serialized(self):
        c1 = self.stack
        c2 = self.get_stack(self)
        before_writing = threading.Event()
        after_writing = threading.Event()
        writing_done = threading.Event()
        c1_save_without_locking_orig = c1.store.save_without_locking

        def c1_save_without_locking():
            before_writing.set()
            c1_save_without_locking_orig()
            after_writing.wait()
        c1.store.save_without_locking = c1_save_without_locking

        def c1_set():
            c1.set('one', 'c1')
            writing_done.set()
        t1 = threading.Thread(target=c1_set)
        self.addCleanup(t1.join)
        self.addCleanup(after_writing.set)
        t1.start()
        before_writing.wait()
        self.assertRaises(errors.LockContention, c2.set, 'one', 'c2')
        self.assertEqual('c1', c1.get('one'))
        after_writing.set()
        writing_done.wait()
        c2.set('one', 'c2')
        self.assertEqual('c2', c2.get('one'))

    def test_read_while_writing(self):
        c1 = self.stack
        ready_to_write = threading.Event()
        do_writing = threading.Event()
        writing_done = threading.Event()
        c1_save_without_locking_orig = c1.store.save_without_locking

        def c1_save_without_locking():
            ready_to_write.set()
            do_writing.wait()
            c1_save_without_locking_orig()
            writing_done.set()
        c1.store.save_without_locking = c1_save_without_locking

        def c1_set():
            c1.set('one', 'c1')
        t1 = threading.Thread(target=c1_set)
        self.addCleanup(t1.join)
        self.addCleanup(do_writing.set)
        t1.start()
        ready_to_write.wait()
        self.assertEqual('c1', c1.get('one'))
        c2 = self.get_stack(self)
        self.assertEqual('1', c2.get('one'))
        do_writing.set()
        writing_done.wait()
        c3 = self.get_stack(self)
        self.assertEqual('c1', c3.get('one'))