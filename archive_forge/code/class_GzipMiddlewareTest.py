import httplib2
from glance.tests import functional
from glance.tests import utils
class GzipMiddlewareTest(functional.FunctionalTest):

    @utils.skip_if_disabled
    def test_gzip_requests(self):
        self.cleanup()
        self.start_servers(**self.__dict__.copy())

        def request(path, headers=None):
            url = 'http://127.0.0.1:%s/v2/%s' % (self.api_port, path)
            http = httplib2.Http()
            return http.request(url, 'GET', headers=headers)
        headers = {'Accept-Encoding': 'identity'}
        response, content = request('images', headers=headers)
        self.assertIsNone(response.get('-content-encoding'))
        headers = {'Accept-Encoding': 'gzip'}
        response, content = request('images', headers=headers)
        self.assertEqual('gzip', response.get('-content-encoding'))
        self.stop_servers()