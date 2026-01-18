import os
import re
from io import BytesIO, StringIO
import yaml
from .. import registry, tests, version_info_formats
from ..bzr.rio import read_stanzas
from ..version_info_formats.format_custom import (CustomVersionInfoBuilder,
from ..version_info_formats.format_python import PythonVersionInfoBuilder
from ..version_info_formats.format_rio import RioVersionInfoBuilder
from ..version_info_formats.format_yaml import YamlVersionInfoBuilder
from . import TestCaseWithTransport
class TestVersionInfoYaml(VersionInfoTestCase):

    def test_yaml_null(self):
        wt = self.make_branch_and_tree('branch')
        bio = StringIO()
        builder = YamlVersionInfoBuilder(wt.branch, working_tree=wt)
        builder.generate(bio)
        val = bio.getvalue()
        self.assertContainsRe(val, 'build-date:')
        self.assertContainsRe(val, "revno: '0'")

    def test_yaml_dotted_revno(self):
        wt = self.create_tree_with_dotted_revno()
        bio = StringIO()
        builder = YamlVersionInfoBuilder(wt.branch, working_tree=wt)
        builder.generate(bio)
        val = bio.getvalue()
        self.assertContainsRe(val, 'revno: 1.1.1')

    def regen_text(self, wt, **kwargs):
        bio = StringIO()
        builder = YamlVersionInfoBuilder(wt.branch, working_tree=wt, **kwargs)
        builder.generate(bio)
        val = bio.getvalue()
        return val

    def test_simple(self):
        wt = self.create_branch()
        val = self.regen_text(wt)
        self.assertContainsRe(val, 'build-date:')
        self.assertContainsRe(val, 'date:')
        self.assertContainsRe(val, "revno: '3'")
        self.assertContainsRe(val, 'revision-id: r3')

    def test_clean(self):
        wt = self.create_branch()
        val = self.regen_text(wt, check_for_clean=True)
        self.assertContainsRe(val, 'clean: true')

    def test_no_clean(self):
        wt = self.create_branch()
        self.build_tree(['branch/c'])
        val = self.regen_text(wt, check_for_clean=True)
        self.assertContainsRe(val, 'clean: false')

    def test_history(self):
        wt = self.create_branch()
        val = self.regen_text(wt, include_revision_history=True)
        self.assertContainsRe(val, 'id: r1')
        self.assertContainsRe(val, 'message: a')
        self.assertContainsRe(val, 'id: r2')
        self.assertContainsRe(val, 'message: ')
        self.assertContainsRe(val, 'id: r3')
        self.assertContainsRe(val, re.escape('message: "\\xE52"'))

    def regen(self, wt, **kwargs):
        bio = StringIO()
        builder = YamlVersionInfoBuilder(wt.branch, working_tree=wt, **kwargs)
        builder.generate(bio)
        bio.seek(0)
        return yaml.safe_load(bio)

    def test_yaml_version_hook(self):

        def update_stanza(rev, stanza):
            stanza['bla'] = 'bloe'
        YamlVersionInfoBuilder.hooks.install_named_hook('revision', update_stanza, None)
        wt = self.create_branch()
        stanza = self.regen(wt)
        self.assertEqual('bloe', stanza['bla'])

    def test_build_date(self):
        wt = self.create_branch()
        stanza = self.regen(wt)
        self.assertTrue('date' in stanza)
        self.assertTrue('build-date' in stanza)
        self.assertEqual('3', stanza['revno'])
        self.assertEqual('r3', stanza['revision-id'])

    def test_not_clean(self):
        wt = self.create_branch()
        self.build_tree(['branch/c'])
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
        self.assertEqual(False, stanza['clean'])

    def test_file_revisions(self):
        wt = self.create_branch()
        self.build_tree(['branch/c'])
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
        file_rev_stanza = stanza['file-revisions']
        self.assertEqual(['', 'a', 'b', 'c'], [r['path'] for r in file_rev_stanza])
        self.assertEqual(['r1', 'r3', 'r2', 'unversioned'], [r['revision'] for r in file_rev_stanza])

    def test_revision_history(self):
        wt = self.create_branch()
        stanza = self.regen(wt, include_revision_history=True)
        revision_stanza = stanza['revisions']
        self.assertEqual(['r1', 'r2', 'r3'], [r['id'] for r in revision_stanza])
        self.assertEqual(['a', 'b', 'å2'], [r['message'] for r in revision_stanza])
        self.assertEqual(3, len([r['date'] for r in revision_stanza]))

    def test_file_revisions_with_rename(self):
        wt = self.create_branch()
        self.build_tree(['branch/a', 'branch/c'])
        wt.add('c')
        wt.rename_one('b', 'd')
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
        file_rev_stanza = stanza['file-revisions']
        self.assertEqual(['', 'a', 'b', 'c', 'd'], [r['path'] for r in file_rev_stanza])
        self.assertEqual(['r1', 'modified', 'renamed to d', 'new', 'renamed from b'], [r['revision'] for r in file_rev_stanza])

    def test_file_revisions_with_removal(self):
        wt = self.create_branch()
        self.build_tree(['branch/a', 'branch/c'])
        wt.add('c')
        wt.rename_one('b', 'd')
        wt.commit('modified', rev_id=b'r4')
        wt.remove(['c', 'd'])
        os.remove('branch/d')
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
        file_rev_stanza = stanza['file-revisions']
        self.assertEqual(['', 'a', 'c', 'd'], [r['path'] for r in file_rev_stanza])
        self.assertEqual(['r1', 'r4', 'unversioned', 'removed'], [r['revision'] for r in file_rev_stanza])

    def test_revision(self):
        wt = self.create_branch()
        self.build_tree(['branch/a', 'branch/c'])
        wt.add('c')
        wt.rename_one('b', 'd')
        stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True, revision_id=wt.last_revision())
        file_rev_stanza = stanza['file-revisions']
        self.assertEqual(['', 'a', 'b'], [r['path'] for r in file_rev_stanza])
        self.assertEqual(['r1', 'r3', 'r2'], [r['revision'] for r in file_rev_stanza])

    def test_no_wt(self):
        wt = self.create_branch()
        self.build_tree(['branch/a', 'branch/c'])
        wt.add('c')
        wt.rename_one('b', 'd')
        bio = StringIO()
        builder = YamlVersionInfoBuilder(wt.branch, working_tree=None, check_for_clean=True, include_file_revisions=True, revision_id=None)
        builder.generate(bio)
        bio.seek(0)
        stanza = yaml.safe_load(bio)
        self.assertEqual([], stanza['file-revisions'])