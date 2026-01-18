import sys
from io import BytesIO
from stat import S_ISDIR
from ...bzr.bzrdir import BzrDirMetaFormat1
from ...bzr.serializer import format_registry as serializer_format_registry
from ...errors import IllegalPath
from ...repository import InterRepository, Repository
from ...tests import TestCase, TestCaseWithTransport
from ...transport import NoSuchFile
from . import xml4
from .bzrdir import BzrDirFormat6
from .repository import (InterWeaveRepo, RepositoryFormat4, RepositoryFormat5,
class TestInterWeaveRepo(TestCaseWithTransport):

    def test_make_repository(self):
        out, err = self.run_bzr('init-shared-repository --format=weave a')
        self.assertEqual(out, 'Standalone tree (format: weave)\nLocation:\n  branch root: a\n')
        self.assertEqual(err, '')

    def test_is_compatible_and_registered(self):
        from ...bzr import knitrepo
        formats = [RepositoryFormat5(), RepositoryFormat6(), RepositoryFormat7()]
        incompatible_formats = [RepositoryFormat4(), knitrepo.RepositoryFormatKnit1()]
        repo_a = self.make_repository('a')
        repo_b = self.make_repository('b')
        is_compatible = InterWeaveRepo.is_compatible
        for source in incompatible_formats:
            repo_a._format = source
            repo_b._format = formats[0]
            self.assertFalse(is_compatible(repo_a, repo_b))
            self.assertFalse(is_compatible(repo_b, repo_a))
        for source in formats:
            repo_a._format = source
            for target in formats:
                repo_b._format = target
                self.assertTrue(is_compatible(repo_a, repo_b))
        self.assertEqual(InterWeaveRepo, InterRepository.get(repo_a, repo_b).__class__)