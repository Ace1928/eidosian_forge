from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
class TestVault(GlacierLayer2Base):

    def setUp(self):
        GlacierLayer2Base.setUp(self)
        self.vault = Vault(self.mock_layer1, FIXTURE_VAULT)

    def test_create_archive_writer(self):
        self.mock_layer1.initiate_multipart_upload.return_value = {'UploadId': 'UPLOADID'}
        writer = self.vault.create_archive_writer(description='stuff')
        self.mock_layer1.initiate_multipart_upload.assert_called_with('examplevault', self.vault.DefaultPartSize, 'stuff')
        self.assertEqual(writer.vault, self.vault)
        self.assertEqual(writer.upload_id, 'UPLOADID')

    def test_delete_vault(self):
        self.vault.delete_archive('archive')
        self.mock_layer1.delete_archive.assert_called_with('examplevault', 'archive')

    def test_initiate_job(self):

        class UTC(tzinfo):
            """UTC"""

            def utcoffset(self, dt):
                return timedelta(0)

            def tzname(self, dt):
                return 'Z'

            def dst(self, dt):
                return timedelta(0)
        self.mock_layer1.initiate_job.return_value = {'JobId': 'job-id'}
        self.vault.retrieve_inventory(start_date=datetime(2014, 1, 1, tzinfo=UTC()), end_date=datetime(2014, 1, 2, tzinfo=UTC()), limit=100)
        self.mock_layer1.initiate_job.assert_called_with('examplevault', {'Type': 'inventory-retrieval', 'InventoryRetrievalParameters': {'StartDate': '2014-01-01T00:00:00Z', 'EndDate': '2014-01-02T00:00:00Z', 'Limit': 100}})

    def test_get_job(self):
        self.mock_layer1.describe_job.return_value = FIXTURE_ARCHIVE_JOB
        job = self.vault.get_job('NkbByEejwEggmBz2fTHgJrg0XBoDfjP4q6iu87-TjhqG6eGoOY9Z8i1_AUyUsuhPAdTqLHy8pTl5nfCFJmDl2yEZONi5L26Omw12vcs01MNGntHEQL8MBfGlqrEXAMPLEArchiveId')
        self.assertEqual(job.action, 'ArchiveRetrieval')

    def test_list_jobs(self):
        self.mock_layer1.list_jobs.return_value = {'JobList': [FIXTURE_ARCHIVE_JOB]}
        jobs = self.vault.list_jobs(False, 'InProgress')
        self.mock_layer1.list_jobs.assert_called_with('examplevault', False, 'InProgress')
        self.assertEqual(jobs[0].archive_id, 'NkbByEejwEggmBz2fTHgJrg0XBoDfjP4q6iu87-TjhqG6eGoOY9Z8i1_AUyUsuhPAdTqLHy8pTl5nfCFJmDl2yEZONi5L26Omw12vcs01MNGntHEQL8MBfGlqrEXAMPLEArchiveId')

    def test_list_all_parts_one_page(self):
        self.mock_layer1.list_parts.return_value = dict(EXAMPLE_PART_LIST_COMPLETE)
        parts_result = self.vault.list_all_parts(sentinel.upload_id)
        expected = [call('examplevault', sentinel.upload_id)]
        self.assertEquals(expected, self.mock_layer1.list_parts.call_args_list)
        self.assertEquals(EXAMPLE_PART_LIST_COMPLETE, parts_result)

    def test_list_all_parts_two_pages(self):
        self.mock_layer1.list_parts.side_effect = [dict(EXAMPLE_PART_LIST_RESULT_PAGE_1), dict(EXAMPLE_PART_LIST_RESULT_PAGE_2)]
        parts_result = self.vault.list_all_parts(sentinel.upload_id)
        expected = [call('examplevault', sentinel.upload_id), call('examplevault', sentinel.upload_id, marker=EXAMPLE_PART_LIST_RESULT_PAGE_1['Marker'])]
        self.assertEquals(expected, self.mock_layer1.list_parts.call_args_list)
        self.assertEquals(EXAMPLE_PART_LIST_COMPLETE, parts_result)

    @patch('boto.glacier.vault.resume_file_upload')
    def test_resume_archive_from_file(self, mock_resume_file_upload):
        part_size = 4
        mock_list_parts = Mock()
        mock_list_parts.return_value = {'PartSizeInBytes': part_size, 'Parts': [{'RangeInBytes': '0-3', 'SHA256TreeHash': '12'}, {'RangeInBytes': '4-6', 'SHA256TreeHash': '34'}]}
        self.vault.list_all_parts = mock_list_parts
        self.vault.resume_archive_from_file(sentinel.upload_id, file_obj=sentinel.file_obj)
        mock_resume_file_upload.assert_called_once_with(self.vault, sentinel.upload_id, part_size, sentinel.file_obj, {0: codecs.decode('12', 'hex_codec'), 1: codecs.decode('34', 'hex_codec')})