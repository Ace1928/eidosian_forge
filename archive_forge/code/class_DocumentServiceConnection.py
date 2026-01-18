import boto.exception
from boto.compat import json
import requests
import boto
class DocumentServiceConnection(object):
    """
    A CloudSearch document service.

    The DocumentServiceConection is used to add, remove and update documents in
    CloudSearch. Commands are uploaded to CloudSearch in SDF (Search Document Format).

    To generate an appropriate SDF, use :func:`add` to add or update documents,
    as well as :func:`delete` to remove documents.

    Once the set of documents is ready to be index, use :func:`commit` to send the
    commands to CloudSearch.

    If there are a lot of documents to index, it may be preferable to split the
    generation of SDF data and the actual uploading into CloudSearch. Retrieve
    the current SDF with :func:`get_sdf`. If this file is the uploaded into S3,
    it can be retrieved back afterwards for upload into CloudSearch using
    :func:`add_sdf_from_s3`.

    The SDF is not cleared after a :func:`commit`. If you wish to continue
    using the DocumentServiceConnection for another batch upload of commands,
    you will need to :func:`clear_sdf` first to stop the previous batch of
    commands from being uploaded again.

    """

    def __init__(self, domain=None, endpoint=None):
        self.domain = domain
        self.endpoint = endpoint
        if not self.endpoint:
            self.endpoint = domain.doc_service_endpoint
        self.documents_batch = []
        self._sdf = None

    def add(self, _id, version, fields, lang='en'):
        """
        Add a document to be processed by the DocumentService

        The document will not actually be added until :func:`commit` is called

        :type _id: string
        :param _id: A unique ID used to refer to this document.

        :type version: int
        :param version: Version of the document being indexed. If a file is
            being reindexed, the version should be higher than the existing one
            in CloudSearch.

        :type fields: dict
        :param fields: A dictionary of key-value pairs to be uploaded .

        :type lang: string
        :param lang: The language code the data is in. Only 'en' is currently
            supported
        """
        d = {'type': 'add', 'id': _id, 'version': version, 'lang': lang, 'fields': fields}
        self.documents_batch.append(d)

    def delete(self, _id, version):
        """
        Schedule a document to be removed from the CloudSearch service

        The document will not actually be scheduled for removal until :func:`commit` is called

        :type _id: string
        :param _id: The unique ID of this document.

        :type version: int
        :param version: Version of the document to remove. The delete will only
            occur if this version number is higher than the version currently
            in the index.
        """
        d = {'type': 'delete', 'id': _id, 'version': version}
        self.documents_batch.append(d)

    def get_sdf(self):
        """
        Generate the working set of documents in Search Data Format (SDF)

        :rtype: string
        :returns: JSON-formatted string of the documents in SDF
        """
        return self._sdf if self._sdf else json.dumps(self.documents_batch)

    def clear_sdf(self):
        """
        Clear the working documents from this DocumentServiceConnection

        This should be used after :func:`commit` if the connection will be reused
        for another set of documents.
        """
        self._sdf = None
        self.documents_batch = []

    def add_sdf_from_s3(self, key_obj):
        """
        Load an SDF from S3

        Using this method will result in documents added through
        :func:`add` and :func:`delete` being ignored.

        :type key_obj: :class:`boto.s3.key.Key`
        :param key_obj: An S3 key which contains an SDF
        """
        self._sdf = key_obj.get_contents_as_string()

    def commit(self):
        """
        Actually send an SDF to CloudSearch for processing

        If an SDF file has been explicitly loaded it will be used. Otherwise,
        documents added through :func:`add` and :func:`delete` will be used.

        :rtype: :class:`CommitResponse`
        :returns: A summary of documents added and deleted
        """
        sdf = self.get_sdf()
        if ': null' in sdf:
            boto.log.error('null value in sdf detected.  This will probably raise 500 error.')
            index = sdf.index(': null')
            boto.log.error(sdf[index - 100:index + 100])
        url = 'http://%s/2011-02-01/documents/batch' % self.endpoint
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=50, max_retries=5)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        r = session.post(url, data=sdf, headers={'Content-Type': 'application/json'})
        return CommitResponse(r, self, sdf)