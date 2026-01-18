from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
class IMAP4ServerTests(IMAP4HelperMixin, TestCase):

    def testCapability(self):
        caps = {}

        def getCaps():

            def gotCaps(c):
                caps.update(c)
                self.server.transport.loseConnection()
            return self.client.getCapabilities().addCallback(gotCaps)
        d1 = self.connected.addCallback(strip(getCaps)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        expected = {b'IMAP4rev1': None, b'NAMESPACE': None, b'IDLE': None}
        return d.addCallback(lambda _: self.assertEqual(expected, caps))

    def testCapabilityWithAuth(self):
        caps = {}
        self.server.challengers[b'CRAM-MD5'] = CramMD5Credentials

        def getCaps():

            def gotCaps(c):
                caps.update(c)
                self.server.transport.loseConnection()
            return self.client.getCapabilities().addCallback(gotCaps)
        d1 = self.connected.addCallback(strip(getCaps)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        expCap = {b'IMAP4rev1': None, b'NAMESPACE': None, b'IDLE': None, b'AUTH': [b'CRAM-MD5']}
        return d.addCallback(lambda _: self.assertEqual(expCap, caps))

    def testLogout(self):
        self.loggedOut = 0

        def logout():

            def setLoggedOut():
                self.loggedOut = 1
            self.client.logout().addCallback(strip(setLoggedOut))
        self.connected.addCallback(strip(logout)).addErrback(self._ebGeneral)
        d = self.loopback()
        return d.addCallback(lambda _: self.assertEqual(self.loggedOut, 1))

    def testNoop(self):
        self.responses = None

        def noop():

            def setResponses(responses):
                self.responses = responses
                self.server.transport.loseConnection()
            self.client.noop().addCallback(setResponses)
        self.connected.addCallback(strip(noop)).addErrback(self._ebGeneral)
        d = self.loopback()
        return d.addCallback(lambda _: self.assertEqual(self.responses, []))

    def testLogin(self):

        def login():
            d = self.client.login(b'testuser', b'password-test')
            d.addCallback(self._cbStopClient)
        d1 = self.connected.addCallback(strip(login)).addErrback(self._ebGeneral)
        d = defer.gatherResults([d1, self.loopback()])
        return d.addCallback(self._cbTestLogin)

    def _cbTestLogin(self, ignored):
        self.assertEqual(self.server.account, SimpleServer.theAccount)
        self.assertEqual(self.server.state, 'auth')

    def testFailedLogin(self):

        def login():
            d = self.client.login(b'testuser', b'wrong-password')
            d.addBoth(self._cbStopClient)
        d1 = self.connected.addCallback(strip(login)).addErrback(self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestFailedLogin)

    def _cbTestFailedLogin(self, ignored):
        self.assertEqual(self.server.account, None)
        self.assertEqual(self.server.state, 'unauth')

    def test_loginWithoutPortal(self):
        """
        Attempting to log into a server that has no L{Portal} results
        in a failed login.
        """
        self.server.portal = None

        def login():
            d = self.client.login(b'testuser', b'wrong-password')
            d.addBoth(self._cbStopClient)
        d1 = self.connected.addCallback(strip(login)).addErrback(self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestFailedLogin)

    def test_nonIAccountAvatar(self):
        """
        The server responds with a C{BAD} response when its portal
        attempts to log a user in with checker that claims to support
        L{IAccount} but returns an an avatar interface that is not
        L{IAccount}.
        """

        def brokenRequestAvatar(*_, **__):
            return ('Not IAccount', 'Not an account', lambda: None)
        self.server.portal.realm.requestAvatar = brokenRequestAvatar

        def login():
            d = self.client.login(b'testuser', b'password-test')
            d.addBoth(self._cbStopClient)
        d1 = self.connected.addCallback(strip(login)).addErrback(self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestFailedLogin)

    def test_loginException(self):
        """
        Any exception raised by L{IMAP4Server.authenticateLogin} that
        is not L{UnauthorizedLogin} is logged results in a C{BAD}
        response.
        """

        class UnexpectedException(Exception):
            """
            An unexpected exception.
            """

        def raisesUnexpectedException(user, passwd):
            raise UnexpectedException('Whoops')
        self.server.authenticateLogin = raisesUnexpectedException

        def login():
            return self.client.login(b'testuser', b'password-test')
        d1 = self.connected.addCallback(strip(login))
        d1.addErrback(self.assertClientFailureMessage, b'Server error: Whoops')

        @d1.addCallback
        def assertErrorLogged(_):
            self.assertTrue(self.flushLoggedErrors(UnexpectedException))
        d1.addErrback(self._ebGeneral)
        d1.addBoth(self._cbStopClient)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestFailedLogin)

    def testLoginRequiringQuoting(self):
        self.server.checker.users = {b'{test}user': b'{test}password'}

        def login():
            d = self.client.login(b'{test}user', b'{test}password')
            d.addErrback(log.err, 'Problem with ' + str(self))
            d.addCallback(self._cbStopClient)
        d1 = self.connected.addCallback(strip(login)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        return d.addCallback(self._cbTestLoginRequiringQuoting)

    def _cbTestLoginRequiringQuoting(self, ignored):
        self.assertEqual(self.server.account, SimpleServer.theAccount)
        self.assertEqual(self.server.state, 'auth')

    def testNamespace(self):
        self.namespaceArgs = None

        def login():
            return self.client.login(b'testuser', b'password-test')

        def namespace():

            def gotNamespace(args):
                self.namespaceArgs = args
                self._cbStopClient(None)
            return self.client.namespace().addCallback(gotNamespace)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(namespace))
        d1.addErrback(self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])

        @d.addCallback
        def assertAllPairsNativeStrings(ignored):
            for namespaces in self.namespaceArgs:
                for pair in namespaces:
                    for value in pair:
                        self.assertIsInstance(value, str)
            return self.namespaceArgs
        d.addCallback(self.assertEqual, [[['', '/']], [], []])
        return d

    def test_mailboxWithoutNamespace(self):
        """
        A mailbox that does not provide L{INamespacePresenter} returns
        empty L{list}s for its personal, shared, and user namespaces.
        """
        self.server.theAccount = AccountWithoutNamespaces(b'testuser')
        self.namespaceArgs = None

        def login():
            return self.client.login(b'testuser', b'password-test')

        def namespace():

            def gotNamespace(args):
                self.namespaceArgs = args
                self._cbStopClient(None)
            return self.client.namespace().addCallback(gotNamespace)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(namespace))
        d1.addErrback(self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.namespaceArgs)
        d.addCallback(self.assertEqual, [[], [], []])
        return d

    def testSelect(self):
        SimpleServer.theAccount.addMailbox('test-mailbox')
        self.selectedArgs = None

        def login():
            return self.client.login(b'testuser', b'password-test')

        def select():

            def selected(args):
                self.selectedArgs = args
                self._cbStopClient(None)
            d = self.client.select('test-mailbox')
            d.addCallback(selected)
            return d
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(select))
        d1.addErrback(self._ebGeneral)
        d2 = self.loopback()
        return defer.gatherResults([d1, d2]).addCallback(self._cbTestSelect)

    def test_selectWithoutMailbox(self):
        """
        A client that selects a mailbox that does not exist receives a
        C{NO} response.
        """

        def login():
            return self.client.login(b'testuser', b'password-test')

        def select():
            return self.client.select('test-mailbox')
        self.connected.addCallback(strip(login))
        self.connected.addCallback(strip(select))
        self.connected.addErrback(self.assertClientFailureMessage, b'No such mailbox')
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        connectionComplete = defer.gatherResults([self.connected, self.loopback()])

        @connectionComplete.addCallback
        def assertNoMailboxSelected(_):
            self.assertIsNone(self.server.mbox)
        return connectionComplete

    def _cbTestSelect(self, ignored):
        mbox = SimpleServer.theAccount.mailboxes['TEST-MAILBOX']
        self.assertEqual(self.server.mbox, mbox)
        self.assertEqual(self.selectedArgs, {'EXISTS': 9, 'RECENT': 3, 'UIDVALIDITY': 42, 'FLAGS': ('\\Flag1', 'Flag2', '\\AnotherSysFlag', 'LastFlag'), 'READ-WRITE': True})

    def test_examine(self):
        """
        L{IMAP4Client.examine} issues an I{EXAMINE} command to the server and
        returns a L{Deferred} which fires with a C{dict} with as many of the
        following keys as the server includes in its response: C{'FLAGS'},
        C{'EXISTS'}, C{'RECENT'}, C{'UNSEEN'}, C{'READ-WRITE'}, C{'READ-ONLY'},
        C{'UIDVALIDITY'}, and C{'PERMANENTFLAGS'}.

        Unfortunately the server doesn't generate all of these so it's hard to
        test the client's handling of them here.  See
        L{IMAP4ClientExamineTests} below.

        See U{RFC 3501<http://www.faqs.org/rfcs/rfc3501.html>}, section 6.3.2,
        for details.
        """
        SimpleServer.theAccount.addMailbox('test-mailbox')
        self.examinedArgs = None

        def login():
            return self.client.login(b'testuser', b'password-test')

        def examine():

            def examined(args):
                self.examinedArgs = args
                self._cbStopClient(None)
            d = self.client.examine('test-mailbox')
            d.addCallback(examined)
            return d
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(examine))
        d1.addErrback(self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestExamine)

    def _cbTestExamine(self, ignored):
        mbox = SimpleServer.theAccount.mailboxes['TEST-MAILBOX']
        self.assertEqual(self.server.mbox, mbox)
        self.assertEqual(self.examinedArgs, {'EXISTS': 9, 'RECENT': 3, 'UIDVALIDITY': 42, 'FLAGS': ('\\Flag1', 'Flag2', '\\AnotherSysFlag', 'LastFlag'), 'READ-WRITE': False})

    def testCreate(self):
        succeed = ('testbox', 'test/box', 'test/', 'test/box/box', 'INBOX')
        fail = ('testbox', 'test/box')

        def cb():
            self.result.append(1)

        def eb(failure):
            self.result.append(0)

        def login():
            return self.client.login(b'testuser', b'password-test')

        def create():
            for name in succeed + fail:
                d = self.client.create(name)
                d.addCallback(strip(cb)).addErrback(eb)
            d.addCallbacks(self._cbStopClient, self._ebGeneral)
        self.result = []
        d1 = self.connected.addCallback(strip(login)).addCallback(strip(create))
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestCreate, succeed, fail)

    def _cbTestCreate(self, ignored, succeed, fail):
        self.assertEqual(self.result, [1] * len(succeed) + [0] * len(fail))
        mbox = sorted(SimpleServer.theAccount.mailboxes)
        answers = sorted(['inbox', 'testbox', 'test/box', 'test', 'test/box/box'])
        self.assertEqual(mbox, [a.upper() for a in answers])

    def testDelete(self):
        SimpleServer.theAccount.addMailbox('delete/me')

        def login():
            return self.client.login(b'testuser', b'password-test')

        def delete():
            return self.client.delete('delete/me')
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(delete), self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.assertEqual(list(SimpleServer.theAccount.mailboxes), []))
        return d

    def testDeleteWithInferiorHierarchicalNames(self):
        """
        Attempting to delete a mailbox with hierarchically inferior
        names fails with an informative error.

        @see: U{https://tools.ietf.org/html/rfc3501#section-6.3.4}

        @return: A L{Deferred} with assertions.
        """
        SimpleServer.theAccount.addMailbox('delete')
        SimpleServer.theAccount.addMailbox('delete/me')

        def login():
            return self.client.login(b'testuser', b'password-test')

        def delete():
            return self.client.delete('delete')

        def assertIMAPException(failure):
            failure.trap(imap4.IMAP4Exception)
            self.assertEqual(str(failure.value), str(b'Name "DELETE" has inferior hierarchical names'))
        loggedIn = self.connected.addCallback(strip(login))
        loggedIn.addCallbacks(strip(delete), self._ebGeneral)
        loggedIn.addErrback(assertIMAPException)
        loggedIn.addCallbacks(self._cbStopClient)
        loopedBack = self.loopback()
        d = defer.gatherResults([loggedIn, loopedBack])
        d.addCallback(lambda _: self.assertEqual(sorted(SimpleServer.theAccount.mailboxes), ['DELETE', 'DELETE/ME']))
        return d

    def testIllegalInboxDelete(self):
        self.stashed = None

        def login():
            return self.client.login(b'testuser', b'password-test')

        def delete():
            return self.client.delete('inbox')

        def stash(result):
            self.stashed = result
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(delete), self._ebGeneral)
        d1.addBoth(stash)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.assertTrue(isinstance(self.stashed, failure.Failure)))
        return d

    def testNonExistentDelete(self):

        def login():
            return self.client.login(b'testuser', b'password-test')

        def delete():
            return self.client.delete('delete/me')

        def deleteFailed(failure):
            self.failure = failure
        self.failure = None
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(delete)).addErrback(deleteFailed)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.assertEqual(str(self.failure.value), str(b'No such mailbox')))
        return d

    def testIllegalDelete(self):
        m = SimpleMailbox()
        m.flags = ('\\Noselect',)
        SimpleServer.theAccount.addMailbox('delete', m)
        SimpleServer.theAccount.addMailbox('delete/me')

        def login():
            return self.client.login(b'testuser', b'password-test')

        def delete():
            return self.client.delete('delete')

        def deleteFailed(failure):
            self.failure = failure
        self.failure = None
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(delete)).addErrback(deleteFailed)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        expected = str(b'Hierarchically inferior mailboxes exist and \\Noselect is set')
        d.addCallback(lambda _: self.assertEqual(str(self.failure.value), expected))
        return d

    def testRename(self):
        SimpleServer.theAccount.addMailbox('oldmbox')

        def login():
            return self.client.login(b'testuser', b'password-test')

        def rename():
            return self.client.rename(b'oldmbox', b'newname')
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(rename), self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.assertEqual(list(SimpleServer.theAccount.mailboxes.keys()), ['NEWNAME']))
        return d

    def testIllegalInboxRename(self):
        self.stashed = None

        def login():
            return self.client.login(b'testuser', b'password-test')

        def rename():
            return self.client.rename('inbox', 'frotz')

        def stash(stuff):
            self.stashed = stuff
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(rename), self._ebGeneral)
        d1.addBoth(stash)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.assertTrue(isinstance(self.stashed, failure.Failure)))
        return d

    def testHierarchicalRename(self):
        SimpleServer.theAccount.create('oldmbox/m1')
        SimpleServer.theAccount.create('oldmbox/m2')

        def login():
            return self.client.login(b'testuser', b'password-test')

        def rename():
            return self.client.rename('oldmbox', 'newname')
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(rename), self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestHierarchicalRename)

    def _cbTestHierarchicalRename(self, ignored):
        mboxes = SimpleServer.theAccount.mailboxes.keys()
        expected = ['newname', 'newname/m1', 'newname/m2']
        mboxes = list(sorted(mboxes))
        self.assertEqual(mboxes, [s.upper() for s in expected])

    def testSubscribe(self):

        def login():
            return self.client.login(b'testuser', b'password-test')

        def subscribe():
            return self.client.subscribe('this/mbox')
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(subscribe), self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.assertEqual(SimpleServer.theAccount.subscriptions, ['THIS/MBOX']))
        return d

    def testUnsubscribe(self):
        SimpleServer.theAccount.subscriptions = ['THIS/MBOX', 'THAT/MBOX']

        def login():
            return self.client.login(b'testuser', b'password-test')

        def unsubscribe():
            return self.client.unsubscribe('this/mbox')
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(unsubscribe), self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.assertEqual(SimpleServer.theAccount.subscriptions, ['THAT/MBOX']))
        return d

    def _listSetup(self, f):
        SimpleServer.theAccount.addMailbox('root/subthing')
        SimpleServer.theAccount.addMailbox('root/another-thing')
        SimpleServer.theAccount.addMailbox('non-root/subthing')

        def login():
            return self.client.login(b'testuser', b'password-test')

        def listed(answers):
            self.listed = answers
        self.listed = None
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(f), self._ebGeneral)
        d1.addCallbacks(listed, self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        return defer.gatherResults([d1, d2]).addCallback(lambda _: self.listed)

    def assertListDelimiterAndMailboxAreStrings(self, results):
        """
        Assert a C{LIST} response's delimiter and mailbox are native
        strings.

        @param results: A list of tuples as returned by
            L{IMAP4Client.list} or L{IMAP4Client.lsub}.
        """
        for result in results:
            self.assertIsInstance(result[1], str, 'delimiter %r is not a str')
            self.assertIsInstance(result[2], str, 'mailbox %r is not a str')
        return results

    def testList(self):

        def mailboxList():
            return self.client.list('root', '%')
        d = self._listSetup(mailboxList)

        @d.addCallback
        def assertListContents(listed):
            expectedContents = [(sorted(SimpleMailbox.flags), '/', 'ROOT/SUBTHING'), (sorted(SimpleMailbox.flags), '/', 'ROOT/ANOTHER-THING')]
            for _ in range(2):
                flags, delimiter, mailbox = listed.pop(0)
                self.assertIn((sorted(flags), delimiter, mailbox), expectedContents)
            self.assertFalse(listed, f'More results than expected: {listed!r}')
        return d

    def testLSub(self):
        SimpleServer.theAccount.subscribe('ROOT/SUBTHING')

        def lsub():
            return self.client.lsub('root', '%')
        d = self._listSetup(lsub)
        d.addCallback(self.assertListDelimiterAndMailboxAreStrings)
        d.addCallback(self.assertEqual, [(SimpleMailbox.flags, '/', 'ROOT/SUBTHING')])
        return d

    def testStatus(self):
        SimpleServer.theAccount.addMailbox('root/subthing')

        def login():
            return self.client.login(b'testuser', b'password-test')

        def status():
            return self.client.status('root/subthing', 'MESSAGES', 'UIDNEXT', 'UNSEEN')

        def statused(result):
            self.statused = result
        self.statused = None
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(status), self._ebGeneral)
        d1.addCallbacks(statused, self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        d.addCallback(lambda _: self.assertEqual(self.statused, {'MESSAGES': 9, 'UIDNEXT': b'10', 'UNSEEN': 4}))
        return d

    def testFailedStatus(self):

        def login():
            return self.client.login(b'testuser', b'password-test')

        def status():
            return self.client.status('root/nonexistent', 'MESSAGES', 'UIDNEXT', 'UNSEEN')

        def statused(result):
            self.statused = result

        def failed(failure):
            self.failure = failure
        self.statused = self.failure = None
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(status), self._ebGeneral)
        d1.addCallbacks(statused, failed)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        return defer.gatherResults([d1, d2]).addCallback(self._cbTestFailedStatus)

    def _cbTestFailedStatus(self, ignored):
        self.assertEqual(self.statused, None)
        self.assertEqual(self.failure.value.args, (b'Could not open mailbox',))

    def testFullAppend(self):
        infile = util.sibpath(__file__, 'rfc822.message')
        SimpleServer.theAccount.addMailbox('root/subthing')

        def login():
            return self.client.login(b'testuser', b'password-test')

        @defer.inlineCallbacks
        def append():
            with open(infile, 'rb') as message:
                result = (yield self.client.append('root/subthing', message, ('\\SEEN', '\\DELETED'), 'Tue, 17 Jun 2003 11:22:16 -0600 (MDT)'))
                defer.returnValue(result)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(append), self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestFullAppend, infile)

    def _cbTestFullAppend(self, ignored, infile):
        mb = SimpleServer.theAccount.mailboxes['ROOT/SUBTHING']
        self.assertEqual(1, len(mb.messages))
        self.assertEqual((['\\SEEN', '\\DELETED'], b'Tue, 17 Jun 2003 11:22:16 -0600 (MDT)', 0), mb.messages[0][1:])
        with open(infile, 'rb') as f:
            self.assertEqual(f.read(), mb.messages[0][0].getvalue())

    def testPartialAppend(self):
        infile = util.sibpath(__file__, 'rfc822.message')
        SimpleServer.theAccount.addMailbox('PARTIAL/SUBTHING')

        def login():
            return self.client.login(b'testuser', b'password-test')

        @defer.inlineCallbacks
        def append():
            with open(infile, 'rb') as message:
                result = (yield self.client.sendCommand(imap4.Command(b'APPEND', networkString('PARTIAL/SUBTHING (\\SEEN) "Right now" {%d}' % (os.path.getsize(infile),)), (), self.client._IMAP4Client__cbContinueAppend, message)))
                defer.returnValue(result)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(append), self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestPartialAppend, infile)

    def _cbTestPartialAppend(self, ignored, infile):
        mb = SimpleServer.theAccount.mailboxes['PARTIAL/SUBTHING']
        self.assertEqual(1, len(mb.messages))
        self.assertEqual((['\\SEEN'], b'Right now', 0), mb.messages[0][1:])
        with open(infile, 'rb') as f:
            self.assertEqual(f.read(), mb.messages[0][0].getvalue())

    def _testCheck(self):
        SimpleServer.theAccount.addMailbox(b'root/subthing')

        def login():
            return self.client.login(b'testuser', b'password-test')

        def select():
            return self.client.select(b'root/subthing')

        def check():
            return self.client.check()
        d = self.connected.addCallback(strip(login))
        d.addCallbacks(strip(select), self._ebGeneral)
        d.addCallbacks(strip(check), self._ebGeneral)
        d.addCallbacks(self._cbStopClient, self._ebGeneral)
        return self.loopback()

    def test_check(self):
        """
        Trigger the L{imap.IMAP4Server._cbSelectWork} callback
        by selecting an mbox.
        """
        return self._testCheck()

    def test_checkFail(self):
        """
        Trigger the L{imap.IMAP4Server._ebSelectWork} errback
        by failing when we select an mbox.
        """

        def failSelect(self, name, rw=1):
            raise imap4.IllegalMailboxEncoding('encoding')

        def checkResponse(ignore):
            failures = self.flushLoggedErrors()
            self.assertEqual(failures[1].value.args[0], b'SELECT failed: Server error')
        self.patch(Account, 'select', failSelect)
        d = self._testCheck()
        return d.addCallback(checkResponse)

    def testClose(self):
        m = SimpleMailbox()
        m.messages = [(b'Message 1', ('\\Deleted', 'AnotherFlag'), None, 0), (b'Message 2', ('AnotherFlag',), None, 1), (b'Message 3', ('\\Deleted',), None, 2)]
        SimpleServer.theAccount.addMailbox('mailbox', m)

        def login():
            return self.client.login(b'testuser', b'password-test')

        def select():
            return self.client.select(b'mailbox')

        def close():
            return self.client.close()
        d = self.connected.addCallback(strip(login))
        d.addCallbacks(strip(select), self._ebGeneral)
        d.addCallbacks(strip(close), self._ebGeneral)
        d.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        return defer.gatherResults([d, d2]).addCallback(self._cbTestClose, m)

    def _cbTestClose(self, ignored, m):
        self.assertEqual(len(m.messages), 1)
        self.assertEqual(m.messages[0], (b'Message 2', ('AnotherFlag',), None, 1))
        self.assertTrue(m.closed)

    def testExpunge(self):
        m = SimpleMailbox()
        m.messages = [(b'Message 1', ('\\Deleted', 'AnotherFlag'), None, 0), (b'Message 2', ('AnotherFlag',), None, 1), (b'Message 3', ('\\Deleted',), None, 2)]
        SimpleServer.theAccount.addMailbox('mailbox', m)

        def login():
            return self.client.login(b'testuser', b'password-test')

        def select():
            return self.client.select('mailbox')

        def expunge():
            return self.client.expunge()

        def expunged(results):
            self.assertFalse(self.server.mbox is None)
            self.results = results
        self.results = None
        d1 = self.connected.addCallback(strip(login))
        d1.addCallbacks(strip(select), self._ebGeneral)
        d1.addCallbacks(strip(expunge), self._ebGeneral)
        d1.addCallbacks(expunged, self._ebGeneral)
        d1.addCallbacks(self._cbStopClient, self._ebGeneral)
        d2 = self.loopback()
        d = defer.gatherResults([d1, d2])
        return d.addCallback(self._cbTestExpunge, m)

    def _cbTestExpunge(self, ignored, m):
        self.assertEqual(len(m.messages), 1)
        self.assertEqual(m.messages[0], (b'Message 2', ('AnotherFlag',), None, 1))
        self.assertEqual(self.results, [0, 2])