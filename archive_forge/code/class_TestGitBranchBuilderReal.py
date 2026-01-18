from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
class TestGitBranchBuilderReal(tests.TestCaseInTempDir):

    def test_create_real_branch(self):
        GitRepo.init('.')
        builder = tests.GitBranchBuilder()
        builder.set_file('foo', b'contents\nfoo\n', False)
        r1 = builder.commit(b'Joe Foo <joe@foo.com>', 'first', timestamp=1194586400)
        mapping = builder.finish()
        self.assertEqual({b'1': b'44411e8e9202177dd19b6599d7a7991059fa3cb4', b'2': b'b0b62e674f67306fddcf72fa888c3b56df100d64'}, mapping)