from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
class CloudSearchConnectionDeletionTest(AWSMockServiceTestCase):
    connection_class = Layer1

    def default_body(self):
        return b'\n<DeleteDomainResponse xmlns="http://cloudsearch.amazonaws.com/doc/2011-02-01">\n  <DeleteDomainResult>\n    <DomainStatus>\n      <SearchPartitionCount>0</SearchPartitionCount>\n      <SearchService>\n        <Arn>arn:aws:cs:us-east-1:1234567890:search/demo</Arn>\n        <Endpoint>search-demo-userdomain.us-east-1.cloudsearch.amazonaws.com</Endpoint>\n      </SearchService>\n      <NumSearchableDocs>0</NumSearchableDocs>\n      <Created>true</Created>\n      <DomainId>1234567890/demo</DomainId>\n      <Processing>false</Processing>\n      <SearchInstanceCount>0</SearchInstanceCount>\n      <DomainName>demo</DomainName>\n      <RequiresIndexDocuments>false</RequiresIndexDocuments>\n      <Deleted>false</Deleted>\n      <DocService>\n        <Arn>arn:aws:cs:us-east-1:1234567890:doc/demo</Arn>\n        <Endpoint>doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com</Endpoint>\n      </DocService>\n    </DomainStatus>\n  </DeleteDomainResult>\n  <ResponseMetadata>\n    <RequestId>00000000-0000-0000-0000-000000000000</RequestId>\n  </ResponseMetadata>\n</DeleteDomainResponse>\n'

    def test_cloudsearch_deletion(self):
        """
        Check that the correct arguments are sent to AWS when creating a
        cloudsearch connection.
        """
        self.set_http_response(status_code=200)
        api_response = self.service_connection.delete_domain('demo')
        self.assert_request_parameters({'Action': 'DeleteDomain', 'DomainName': 'demo', 'Version': '2011-02-01'})