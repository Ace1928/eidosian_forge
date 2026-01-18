from twisted.python import usage
from twisted.trial import unittest
class SubCommandTests(unittest.TestCase):
    """
    Test L{usage.Options.parseOptions} for options with subcommands.
    """

    def test_simpleSubcommand(self):
        """
        A subcommand is recognized.
        """
        o = SubCommandOptions()
        o.parseOptions(['--europian-swallow', 'inquisition'])
        self.assertTrue(o['europian-swallow'])
        self.assertEqual(o.subCommand, 'inquisition')
        self.assertIsInstance(o.subOptions, InquisitionOptions)
        self.assertFalse(o.subOptions['expect'])
        self.assertEqual(o.subOptions['torture-device'], 'comfy-chair')

    def test_subcommandWithFlagsAndOptions(self):
        """
        Flags and options of a subcommand are assigned.
        """
        o = SubCommandOptions()
        o.parseOptions(['inquisition', '--expect', '--torture-device=feather'])
        self.assertFalse(o['europian-swallow'])
        self.assertEqual(o.subCommand, 'inquisition')
        self.assertIsInstance(o.subOptions, InquisitionOptions)
        self.assertTrue(o.subOptions['expect'])
        self.assertEqual(o.subOptions['torture-device'], 'feather')

    def test_subcommandAliasWithFlagsAndOptions(self):
        """
        Flags and options of a subcommand alias are assigned.
        """
        o = SubCommandOptions()
        o.parseOptions(['inquest', '--expect', '--torture-device=feather'])
        self.assertFalse(o['europian-swallow'])
        self.assertEqual(o.subCommand, 'inquisition')
        self.assertIsInstance(o.subOptions, InquisitionOptions)
        self.assertTrue(o.subOptions['expect'])
        self.assertEqual(o.subOptions['torture-device'], 'feather')

    def test_anotherSubcommandWithFlagsAndOptions(self):
        """
        Flags and options of another subcommand are assigned.
        """
        o = SubCommandOptions()
        o.parseOptions(['holyquest', '--for-grail'])
        self.assertFalse(o['europian-swallow'])
        self.assertEqual(o.subCommand, 'holyquest')
        self.assertIsInstance(o.subOptions, HolyQuestOptions)
        self.assertFalse(o.subOptions['horseback'])
        self.assertTrue(o.subOptions['for-grail'])

    def test_noSubcommand(self):
        """
        If no subcommand is specified and no default subcommand is assigned,
        a subcommand will not be implied.
        """
        o = SubCommandOptions()
        o.parseOptions(['--europian-swallow'])
        self.assertTrue(o['europian-swallow'])
        self.assertIsNone(o.subCommand)
        self.assertFalse(hasattr(o, 'subOptions'))

    def test_defaultSubcommand(self):
        """
        Flags and options in the default subcommand are assigned.
        """
        o = SubCommandOptions()
        o.defaultSubCommand = 'inquest'
        o.parseOptions(['--europian-swallow'])
        self.assertTrue(o['europian-swallow'])
        self.assertEqual(o.subCommand, 'inquisition')
        self.assertIsInstance(o.subOptions, InquisitionOptions)
        self.assertFalse(o.subOptions['expect'])
        self.assertEqual(o.subOptions['torture-device'], 'comfy-chair')

    def test_subCommandParseOptionsHasParent(self):
        """
        The parseOptions method from the Options object specified for the
        given subcommand is called.
        """

        class SubOpt(usage.Options):

            def parseOptions(self, *a, **kw):
                self.sawParent = self.parent
                usage.Options.parseOptions(self, *a, **kw)

        class Opt(usage.Options):
            subCommands = [('foo', 'f', SubOpt, 'bar')]
        o = Opt()
        o.parseOptions(['foo'])
        self.assertTrue(hasattr(o.subOptions, 'sawParent'))
        self.assertEqual(o.subOptions.sawParent, o)

    def test_subCommandInTwoPlaces(self):
        """
        The .parent pointer is correct even when the same Options class is
        used twice.
        """

        class SubOpt(usage.Options):
            pass

        class OptFoo(usage.Options):
            subCommands = [('foo', 'f', SubOpt, 'quux')]

        class OptBar(usage.Options):
            subCommands = [('bar', 'b', SubOpt, 'quux')]
        oFoo = OptFoo()
        oFoo.parseOptions(['foo'])
        oBar = OptBar()
        oBar.parseOptions(['bar'])
        self.assertTrue(hasattr(oFoo.subOptions, 'parent'))
        self.assertTrue(hasattr(oBar.subOptions, 'parent'))
        self.failUnlessIdentical(oFoo.subOptions.parent, oFoo)
        self.failUnlessIdentical(oBar.subOptions.parent, oBar)