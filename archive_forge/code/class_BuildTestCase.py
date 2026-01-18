import os
import unittest
import warnings
class BuildTestCase(support.TempdirManager, unittest.TestCase):

    def test_formats(self):
        dist = self.create_dist()[1]
        cmd = bdist(dist)
        cmd.formats = ['tar']
        cmd.ensure_finalized()
        self.assertEqual(cmd.formats, ['tar'])
        formats = ['bztar', 'gztar', 'rpm', 'tar', 'xztar', 'zip', 'ztar']
        found = sorted(cmd.format_command)
        self.assertEqual(found, formats)

    def test_skip_build(self):
        dist = self.create_dist()[1]
        cmd = bdist(dist)
        cmd.skip_build = 1
        cmd.ensure_finalized()
        dist.command_obj['bdist'] = cmd
        for name in ['bdist_dumb']:
            subcmd = cmd.get_finalized_command(name)
            if getattr(subcmd, '_unsupported', False):
                continue
            self.assertTrue(subcmd.skip_build, '%s should take --skip-build from bdist' % name)