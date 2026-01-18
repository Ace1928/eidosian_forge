import aiohttp  # type: ignore
from aioresponses import aioresponses, core  # type: ignore
import mock
import pytest  # type: ignore
from tests_async.transport import async_compliance
import google.auth._credentials_async
from google.auth.transport import _aiohttp_requests as aiohttp_requests
import google.auth.transport._mtls_helper
class TestCombinedResponse:

    @pytest.mark.asyncio
    async def test__is_compressed(self):
        response = core.CallbackResult(headers={'Content-Encoding': 'gzip'})
        combined_response = aiohttp_requests._CombinedResponse(response)
        compressed = combined_response._is_compressed()
        assert compressed

    def test__is_compressed_not(self):
        response = core.CallbackResult(headers={'Content-Encoding': 'not'})
        combined_response = aiohttp_requests._CombinedResponse(response)
        compressed = combined_response._is_compressed()
        assert not compressed

    @pytest.mark.asyncio
    async def test_raw_content(self):
        mock_response = mock.AsyncMock()
        mock_response.content.read.return_value = mock.sentinel.read
        combined_response = aiohttp_requests._CombinedResponse(response=mock_response)
        raw_content = await combined_response.raw_content()
        assert raw_content == mock.sentinel.read
        combined_response._raw_content = mock.sentinel.stored_raw
        raw_content = await combined_response.raw_content()
        assert raw_content == mock.sentinel.stored_raw

    @pytest.mark.asyncio
    async def test_content(self):
        mock_response = mock.AsyncMock()
        mock_response.content.read.return_value = mock.sentinel.read
        combined_response = aiohttp_requests._CombinedResponse(response=mock_response)
        content = await combined_response.content()
        assert content == mock.sentinel.read

    @mock.patch('google.auth.transport._aiohttp_requests.urllib3.response.MultiDecoder.decompress', return_value='decompressed', autospec=True)
    @pytest.mark.asyncio
    async def test_content_compressed(self, urllib3_mock):
        rm = core.RequestMatch('url', headers={'Content-Encoding': 'gzip'}, payload='compressed')
        response = await rm.build_response(core.URL('url'))
        combined_response = aiohttp_requests._CombinedResponse(response=response)
        content = await combined_response.content()
        urllib3_mock.assert_called_once()
        assert content == 'decompressed'