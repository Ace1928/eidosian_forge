from novaclient.tests.functional import base
class ResourceChecker(base.ClientTestBase):

    def runTest(self):
        pass

    def check(self):
        self.setUp()
        print('$ nova list --all-tenants')
        print(self.nova('list', params='--all-tenants'))
        print('\n')