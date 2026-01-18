import os
from io import StringIO
from typing import Sequence, Type
from unittest import skipIf
from zope.interface import Interface
from twisted import plugin
from twisted.cred import checkers, credentials, error, strcred
from twisted.plugins import cred_anonymous, cred_file, cred_unix
from twisted.python import usage
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
class LimitingInterfacesTests(TestCase):
    """
    Tests functionality that allows an application to limit the
    credential interfaces it can support. For the purposes of this
    test, we use IUsernameHashedPassword, although this will never
    really be used by the command line.

    (I have, to date, not thought of a half-decent way for a user to
    specify a hash algorithm via the command-line. Nor do I think it's
    very useful.)

    I should note that, at first, this test is counter-intuitive,
    because we're using the checker with a pre-defined hash function
    as the 'bad' checker. See the documentation for
    L{twisted.cred.checkers.FilePasswordDB.hash} for more details.
    """

    def setUp(self):
        self.filename = self.mktemp()
        with open(self.filename, 'wb') as f:
            f.write(b'admin:asdf\nalice:foo\n')
        self.goodChecker = checkers.FilePasswordDB(self.filename)
        self.badChecker = checkers.FilePasswordDB(self.filename, hash=self._hash)
        self.anonChecker = checkers.AllowAnonymousAccess()

    def _hash(self, networkUsername, networkPassword, storedPassword):
        """
        A dumb hash that doesn't really do anything.
        """
        return networkPassword

    def test_supportsInterface(self):
        """
        The supportsInterface method behaves appropriately.
        """
        options = OptionsForUsernamePassword()
        self.assertTrue(options.supportsInterface(credentials.IUsernamePassword))
        self.assertFalse(options.supportsInterface(credentials.IAnonymous))
        self.assertRaises(strcred.UnsupportedInterfaces, options.addChecker, self.anonChecker)

    def test_supportsAllInterfaces(self):
        """
        The supportsInterface method behaves appropriately
        when the supportedInterfaces attribute is None.
        """
        options = OptionsSupportsAllInterfaces()
        self.assertTrue(options.supportsInterface(credentials.IUsernamePassword))
        self.assertTrue(options.supportsInterface(credentials.IAnonymous))

    def test_supportsCheckerFactory(self):
        """
        The supportsCheckerFactory method behaves appropriately.
        """
        options = OptionsForUsernamePassword()
        fileCF = cred_file.theFileCheckerFactory
        anonCF = cred_anonymous.theAnonymousCheckerFactory
        self.assertTrue(options.supportsCheckerFactory(fileCF))
        self.assertFalse(options.supportsCheckerFactory(anonCF))

    def test_canAddSupportedChecker(self):
        """
        When addChecker is called with a checker that implements at least one
        of the interfaces our application supports, it is successful.
        """
        options = OptionsForUsernamePassword()
        options.addChecker(self.goodChecker)
        iface = options.supportedInterfaces[0]
        self.assertIdentical(options['credInterfaces'][iface][0], self.goodChecker)
        self.assertIdentical(options['credCheckers'][0], self.goodChecker)
        self.assertEqual(len(options['credInterfaces'][iface]), 1)
        self.assertEqual(len(options['credCheckers']), 1)

    def test_failOnAddingUnsupportedChecker(self):
        """
        When addChecker is called with a checker that does not implement any
        supported interfaces, it fails.
        """
        options = OptionsForUsernameHashedPassword()
        self.assertRaises(strcred.UnsupportedInterfaces, options.addChecker, self.badChecker)

    def test_unsupportedInterfaceError(self):
        """
        The C{--auth} command line raises an exception when it
        gets a checker we don't support.
        """
        options = OptionsSupportsNoInterfaces()
        authType = cred_anonymous.theAnonymousCheckerFactory.authType
        self.assertRaises(usage.UsageError, options.parseOptions, ['--auth', authType])

    def test_helpAuthLimitsOutput(self):
        """
        C{--help-auth} will only list checkers that purport to
        supply at least one of the credential interfaces our
        application can use.
        """
        options = OptionsForUsernamePassword()
        for factory in options._checkerFactoriesForOptHelpAuth():
            invalid = True
            for interface in factory.credentialInterfaces:
                if options.supportsInterface(interface):
                    invalid = False
            if invalid:
                raise strcred.UnsupportedInterfaces()

    def test_helpAuthTypeLimitsOutput(self):
        """
        C{--help-auth-type} will display a warning if you get
        help for an authType that does not supply at least one of the
        credential interfaces our application can use.
        """
        options = OptionsForUsernamePassword()
        invalidFactory = None
        for factory in strcred.findCheckerFactories():
            if not options.supportsCheckerFactory(factory):
                invalidFactory = factory
                break
        self.assertNotIdentical(invalidFactory, None)
        newStdout = StringIO()
        options.authOutput = newStdout
        self.assertRaises(SystemExit, options.parseOptions, ['--help-auth-type', 'anonymous'])
        self.assertIn(strcred.notSupportedWarning, newStdout.getvalue())