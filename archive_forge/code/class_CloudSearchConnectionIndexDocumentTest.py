from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
class CloudSearchConnectionIndexDocumentTest(AWSMockServiceTestCase):
    connection_class = Layer1

    def default_body(self):
        return b'\n<IndexDocumentsResponse xmlns="http://cloudsearch.amazonaws.com/doc/2011-02-01">\n  <IndexDocumentsResult>\n    <FieldNames>\n      <member>average_score</member>\n      <member>brand_id</member>\n      <member>colors</member>\n      <member>context</member>\n      <member>context_owner</member>\n      <member>created_at</member>\n      <member>creator_id</member>\n      <member>description</member>\n      <member>file_size</member>\n      <member>format</member>\n      <member>has_logo</member>\n      <member>has_messaging</member>\n      <member>height</member>\n      <member>image_id</member>\n      <member>ingested_from</member>\n      <member>is_advertising</member>\n      <member>is_photo</member>\n      <member>is_reviewed</member>\n      <member>modified_at</member>\n      <member>subject_date</member>\n      <member>tags</member>\n      <member>title</member>\n      <member>width</member>\n    </FieldNames>\n  </IndexDocumentsResult>\n  <ResponseMetadata>\n    <RequestId>eb2b2390-6bbd-11e2-ab66-93f3a90dcf2a</RequestId>\n  </ResponseMetadata>\n</IndexDocumentsResponse>\n'

    def test_cloudsearch_index_documents(self):
        """
        Check that the correct arguments are sent to AWS when indexing a
        domain.
        """
        self.set_http_response(status_code=200)
        api_response = self.service_connection.index_documents('demo')
        self.assert_request_parameters({'Action': 'IndexDocuments', 'DomainName': 'demo', 'Version': '2011-02-01'})

    def test_cloudsearch_index_documents_resp(self):
        """
        Check that the AWS response is being parsed correctly when indexing a
        domain.
        """
        self.set_http_response(status_code=200)
        api_response = self.service_connection.index_documents('demo')
        self.assertEqual(api_response, ['average_score', 'brand_id', 'colors', 'context', 'context_owner', 'created_at', 'creator_id', 'description', 'file_size', 'format', 'has_logo', 'has_messaging', 'height', 'image_id', 'ingested_from', 'is_advertising', 'is_photo', 'is_reviewed', 'modified_at', 'subject_date', 'tags', 'title', 'width'])