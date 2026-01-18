from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
class FakeHTTPClient(base_client.HTTPClient):

    def __init__(self, version_header=None, **kwargs):
        self.username = 'username'
        self.password = 'password'
        self.auth_url = 'auth_url'
        self.callstack = []
        self.management_url = 'http://10.0.2.15:8776/v2/fake'
        self.osapi_max_limit = 1000
        self.marker = None
        self.version_header = version_header

    def _cs_request(self, url, method, **kwargs):
        if method in ['GET', 'DELETE']:
            assert 'body' not in kwargs
        elif method == 'PUT':
            assert 'body' in kwargs
        args = urlparse.parse_qsl(urlparse.urlparse(url)[4])
        kwargs.update(args)
        url_split = url.rsplit('?', 1)
        munged_url = url_split[0]
        if len(url_split) > 1:
            parameters = url_split[1]
            if 'marker' in parameters:
                self.marker = int(parameters.rsplit('marker=', 1)[1])
            else:
                self.marker = None
        else:
            self.marker = None
        munged_url = munged_url.strip('/').replace('/', '_').replace('.', '_')
        munged_url = munged_url.replace('-', '_')
        callback = '%s_%s' % (method.lower(), munged_url)
        if not hasattr(self, callback):
            raise AssertionError('Called unknown API method: %s %s, expected fakes method name: %s' % (method, url, callback))
        self.callstack.append((method, url, kwargs.get('body', None)))
        status, headers, body = getattr(self, callback)(**kwargs)
        headers['x-openstack-request-id'] = REQUEST_ID
        if self.version_header:
            headers['OpenStack-API-version'] = self.version_header
        r = utils.TestResponse({'status_code': status, 'text': body, 'headers': headers})
        return (r, body)

    def get_volume_api_version_from_endpoint(self):
        magic_tuple = urlparse.urlsplit(self.management_url)
        scheme, netloc, path, query, frag = magic_tuple
        return path.lstrip('/').split('/')[0][1:]

    def get_snapshots_detail(self, **kw):
        if kw.get('with_count', False):
            return (200, {}, {'snapshots': [_stub_snapshot()], 'count': 1})
        return (200, {}, {'snapshots': [_stub_snapshot()]})

    def get_snapshots_1234(self, **kw):
        return (200, {}, {'snapshot': _stub_snapshot(id='1234')})

    def get_snapshots_5678(self, **kw):
        return (200, {}, {'snapshot': _stub_snapshot(id='5678')})

    def post_snapshots(self, **kw):
        metadata = kw['body']['snapshot'].get('metadata', None)
        snapshot = _stub_snapshot(id='1234', volume_id='1234')
        if snapshot is not None:
            snapshot.update({'metadata': metadata})
        return (202, {}, {'snapshot': snapshot})

    def put_snapshots_1234(self, **kw):
        snapshot = _stub_snapshot(id='1234')
        snapshot.update(kw['body']['snapshot'])
        return (200, {}, {'snapshot': snapshot})

    def post_snapshots_1234_action(self, body, **kw):
        _body = None
        resp = 202
        assert len(list(body)) == 1
        action = list(body)[0]
        if action == 'os-reset_status':
            assert 'status' in body['os-reset_status']
        elif action == 'os-update_snapshot_status':
            assert 'status' in body['os-update_snapshot_status']
        elif action == 'os-force_delete':
            assert body[action] is None
        elif action == 'os-unmanage':
            assert body[action] is None
        else:
            raise AssertionError('Unexpected action: %s' % action)
        return (resp, {}, _body)

    def post_snapshots_5678_action(self, body, **kw):
        return self.post_snapshots_1234_action(body, **kw)

    def delete_snapshots_1234(self, **kw):
        return (202, {}, {})

    def delete_snapshots_5678(self, **kw):
        return (202, {}, {})

    def put_volumes_1234(self, **kw):
        volume = _stub_volume(id='1234')
        volume.update(kw['body']['volume'])
        return (200, {}, {'volume': volume})

    def get_volumes(self, **kw):
        if self.marker == 1234:
            return (200, {}, {'volumes': [{'id': 5678, 'name': 'sample-volume2'}]})
        elif self.osapi_max_limit == 1:
            return (200, {}, {'volumes': [{'id': 1234, 'name': 'sample-volume'}], 'volumes_links': [{'href': '/volumes?limit=1&marker=1234', 'rel': 'next'}]})
        else:
            return (200, {}, {'volumes': [{'id': 1234, 'name': 'sample-volume'}, {'id': 5678, 'name': 'sample-volume2'}]})

    def get_volumes_detail(self, **kw):
        if kw.get('with_count', False):
            return (200, {}, {'volumes': [_stub_volume(id=kw.get('id', 1234))], 'count': 1})
        return (200, {}, {'volumes': [_stub_volume(id=kw.get('id', 1234))]})

    def get_volumes_1234(self, **kw):
        r = {'volume': self.get_volumes_detail(id=1234)[2]['volumes'][0]}
        return (200, {}, r)

    def get_volumes_5678(self, **kw):
        r = {'volume': self.get_volumes_detail(id=5678)[2]['volumes'][0]}
        return (200, {}, r)

    def get_volumes_1234_metadata(self, **kw):
        r = {'metadata': {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}}
        return (200, {}, r)

    def get_volumes_1234_encryption(self, **kw):
        r = {'encryption_key_id': 'id'}
        return (200, {}, r)

    def post_volumes_1234_action(self, body, **kw):
        _body = None
        resp = 202
        assert len(list(body)) == 1
        action = list(body)[0]
        if action == 'os-attach':
            keys = sorted(list(body[action]))
            assert keys == ['instance_uuid', 'mode', 'mountpoint'] or keys == ['host_name', 'mode', 'mountpoint']
        elif action == 'os-detach':
            assert list(body[action]) == ['attachment_id']
        elif action == 'os-reserve':
            assert body[action] is None
        elif action == 'os-unreserve':
            assert body[action] is None
        elif action == 'os-initialize_connection':
            assert list(body[action]) == ['connector']
            return (202, {}, {'connection_info': {'foos': 'bars'}})
        elif action == 'os-terminate_connection':
            assert list(body[action]) == ['connector']
        elif action == 'os-begin_detaching':
            assert body[action] is None
        elif action == 'os-roll_detaching':
            assert body[action] is None
        elif action == 'os-reset_status':
            assert 'status' or 'attach_status' or 'migration_status' in body[action]
        elif action == 'os-extend':
            assert list(body[action]) == ['new_size']
        elif action == 'os-migrate_volume':
            assert 'host' in body[action]
            assert 'force_host_copy' in body[action]
        elif action == 'os-update_readonly_flag':
            assert list(body[action]) == ['readonly']
        elif action == 'os-retype':
            assert 'new_type' in body[action]
        elif action == 'os-set_bootable':
            assert list(body[action]) == ['bootable']
        elif action == 'os-unmanage':
            assert body[action] is None
        elif action == 'os-set_image_metadata':
            assert list(body[action]) == ['metadata']
        elif action == 'os-unset_image_metadata':
            assert 'key' in body[action]
        elif action == 'os-show_image_metadata':
            assert body[action] is None
        elif action == 'os-volume_upload_image':
            assert 'image_name' in body[action]
            _body = body
        elif action == 'revert':
            assert 'snapshot_id' in body[action]
        elif action == 'os-reimage':
            assert 'image_id' in body[action]
        elif action == 'os-extend_volume_completion':
            assert 'error' in body[action]
        else:
            raise AssertionError('Unexpected action: %s' % action)
        return (resp, {}, _body)

    def get_volumes_fake(self, **kw):
        r = {'volume': self.get_volumes_detail(id='fake')[2]['volumes'][0]}
        return (200, {}, r)

    def post_volumes_fake_action(self, body, **kw):
        _body = None
        resp = 202
        return (resp, {}, _body)

    def post_volumes_5678_action(self, body, **kw):
        return self.post_volumes_1234_action(body, **kw)

    def post_volumes(self, **kw):
        size = kw['body']['volume'].get('size', 1)
        volume = _stub_volume(id='1234', size=size)
        return (202, {}, {'volume': volume})

    def delete_volumes_1234(self, **kw):
        return (202, {}, None)

    def delete_volumes_5678(self, **kw):
        return (202, {}, None)

    def get_consistencygroups_detail(self, **kw):
        return (200, {}, {'consistencygroups': [_stub_consistencygroup(id='1234'), _stub_consistencygroup(id='4567')]})

    def get_consistencygroups(self, **kw):
        return (200, {}, {'consistencygroups': [_stub_consistencygroup(detailed=False, id='1234'), _stub_consistencygroup(detailed=False, id='4567')]})

    def get_consistencygroups_1234(self, **kw):
        return (200, {}, {'consistencygroup': _stub_consistencygroup(id='1234')})

    def post_consistencygroups(self, **kw):
        return (202, {}, {'consistencygroup': {}})

    def put_consistencygroups_1234(self, **kw):
        return (200, {}, {'consistencygroup': {}})

    def post_consistencygroups_1234_delete(self, **kw):
        return (202, {}, {})

    def post_consistencygroups_create_from_src(self, **kw):
        return (200, {}, {'consistencygroup': _stub_consistencygroup(id='1234', cgsnapshot_id='1234')})

    def get_cgsnapshots_detail(self, **kw):
        return (200, {}, {'cgsnapshots': [_stub_cgsnapshot(id='1234'), _stub_cgsnapshot(id='4567')]})

    def get_cgsnapshots(self, **kw):
        return (200, {}, {'cgsnapshots': [_stub_cgsnapshot(detailed=False, id='1234'), _stub_cgsnapshot(detailed=False, id='4567')]})

    def get_cgsnapshots_1234(self, **kw):
        return (200, {}, {'cgsnapshot': _stub_cgsnapshot(id='1234')})

    def post_cgsnapshots(self, **kw):
        return (202, {}, {'cgsnapshot': {}})

    def put_cgsnapshots_1234(self, **kw):
        return (200, {}, {'cgsnapshot': {}})

    def delete_cgsnapshots_1234(self, **kw):
        return (202, {}, {})

    def get_os_quota_sets_test(self, **kw):
        return (200, {}, {'quota_set': {'tenant_id': 'test', 'metadata_items': [], 'volumes': 1, 'snapshots': 1, 'gigabytes': 1, 'backups': 1, 'backup_gigabytes': 1, 'per_volume_gigabytes': 1}})

    def get_os_quota_sets_test_defaults(self):
        return (200, {}, {'quota_set': {'tenant_id': 'test', 'metadata_items': [], 'volumes': 1, 'snapshots': 1, 'gigabytes': 1, 'backups': 1, 'backup_gigabytes': 1, 'per_volume_gigabytes': 1}})

    def put_os_quota_sets_test(self, body, **kw):
        assert list(body) == ['quota_set']
        fakes.assert_has_keys(body['quota_set'], required=['tenant_id'])
        return (200, {}, {'quota_set': {'tenant_id': 'test', 'metadata_items': [], 'volumes': 2, 'snapshots': 2, 'gigabytes': 1, 'backups': 1, 'backup_gigabytes': 1, 'per_volume_gigabytes': 1}})

    def delete_os_quota_sets_1234(self, **kw):
        return (200, {}, {})

    def delete_os_quota_sets_test(self, **kw):
        return (200, {}, {})

    def get_os_quota_class_sets_test(self, **kw):
        return (200, {}, {'quota_class_set': {'class_name': 'test', 'volumes': 1, 'snapshots': 1, 'gigabytes': 1, 'backups': 1, 'backup_gigabytes': 1, 'per_volume_gigabytes': 1}})

    def put_os_quota_class_sets_test(self, body, **kw):
        assert list(body) == ['quota_class_set']
        fakes.assert_has_keys(body['quota_class_set'])
        return (200, {}, {'quota_class_set': {'volumes': 2, 'snapshots': 2, 'gigabytes': 1, 'backups': 1, 'backup_gigabytes': 1, 'per_volume_gigabytes': 1}})

    def get_types(self, **kw):
        return (200, {}, {'volume_types': [{'id': 1, 'name': 'test-type-1', 'description': 'test_type-1-desc', 'extra_specs': {}}, {'id': 2, 'name': 'test-type-2', 'description': 'test_type-2-desc', 'extra_specs': {}}]})

    def get_types_1(self, **kw):
        return (200, {}, {'volume_type': {'id': 1, 'name': 'test-type-1', 'description': 'test_type-1-desc', 'extra_specs': {'key': 'value'}}})

    def get_types_2(self, **kw):
        return (200, {}, {'volume_type': {'id': 2, 'name': 'test-type-2', 'description': 'test_type-2-desc', 'extra_specs': {}}})

    def get_types_3(self, **kw):
        return (200, {}, {'volume_type': {'id': 3, 'name': 'test-type-3', 'description': 'test_type-3-desc', 'extra_specs': {}, 'os-volume-type-access:is_public': False}})

    def get_types_default(self, **kw):
        return self.get_types_1()

    def post_types(self, body, **kw):
        return (202, {}, {'volume_type': {'id': 3, 'name': 'test-type-3', 'description': 'test_type-3-desc', 'extra_specs': {}}})

    def post_types_3_action(self, body, **kw):
        _body = None
        resp = 202
        assert len(list(body)) == 1
        action = list(body)[0]
        if action == 'addProjectAccess':
            assert 'project' in body['addProjectAccess']
        elif action == 'removeProjectAccess':
            assert 'project' in body['removeProjectAccess']
        else:
            raise AssertionError('Unexpected action: %s' % action)
        return (resp, {}, _body)

    def post_types_1_extra_specs(self, body, **kw):
        assert list(body) == ['extra_specs']
        return (200, {}, {'extra_specs': {'k': 'v'}})

    def delete_types_1_extra_specs_k(self, **kw):
        return (204, {}, None)

    def delete_types_1_extra_specs_m(self, **kw):
        return (204, {}, None)

    def delete_types_1(self, **kw):
        return (202, {}, None)

    def delete_types_3_extra_specs_k(self, **kw):
        return (204, {}, None)

    def delete_types_3(self, **kw):
        return (202, {}, None)

    def put_types_1(self, **kw):
        return self.get_types_1()

    def put_types_3(self, **kw):
        return (200, {}, {'volume_type': {'id': 3, 'name': 'test-type-2', 'description': 'test_type-3-desc', 'is_public': True, 'extra_specs': {}}})

    def get_types_3_os_volume_type_access(self, **kw):
        return (200, {}, {'volume_type_access': [_stub_type_access()]})

    def get_types_1_encryption(self, **kw):
        return (200, {}, {'id': 1, 'volume_type_id': 1, 'provider': 'test', 'cipher': 'test', 'key_size': 1, 'control_location': 'front-end'})

    def get_types_2_encryption(self, **kw):
        return (200, {}, {})

    def post_types_2_encryption(self, body, **kw):
        return (200, {}, {'encryption': body})

    def put_types_1_encryption_provider(self, body, **kw):
        get_body = self.get_types_1_encryption()[2]
        for k, v in body.items():
            if k in get_body.keys():
                get_body.update([(k, v)])
        return (200, {}, get_body)

    def delete_types_1_encryption_provider(self, **kw):
        return (202, {}, None)

    def delete_volumes_1234_metadata_test_key(self, **kw):
        return (204, {}, None)

    def delete_volumes_1234_metadata_key1(self, **kw):
        return (204, {}, None)

    def delete_volumes_1234_metadata_key2(self, **kw):
        return (204, {}, None)

    def post_volumes_1234_metadata(self, **kw):
        return (204, {}, {'metadata': {'test_key': 'test_value'}})

    def get_extensions(self, **kw):
        exts = [{'alias': 'FAKE-1', 'description': 'Fake extension number 1', 'links': [], 'name': 'Fake1', 'namespace': 'http://docs.openstack.org//ext/fake1/api/v1.1', 'updated': '2011-06-09T00:00:00+00:00'}, {'alias': 'FAKE-2', 'description': 'Fake extension number 2', 'links': [], 'name': 'Fake2', 'namespace': 'http://docs.openstack.org//ext/fake1/api/v1.1', 'updated': '2011-06-09T00:00:00+00:00'}]
        return (200, {}, {'extensions': exts})

    def get_backups_76a17945_3c6f_435c_975b_b5685db10b62(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        backup1 = '76a17945-3c6f-435c-975b-b5685db10b62'
        return (200, {}, {'backup': _stub_backup_full(backup1, base_uri, tenant_id)})

    def get_backups_1234(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        backup1 = '1234'
        return (200, {}, {'backup': _stub_backup_full(backup1, base_uri, tenant_id)})

    def get_backups_5678(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        backup1 = '5678'
        return (200, {}, {'backup': _stub_backup_full(backup1, base_uri, tenant_id)})

    def get_backups_detail(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        backup1 = '76a17945-3c6f-435c-975b-b5685db10b62'
        backup2 = 'd09534c6-08b8-4441-9e87-8976f3a8f699'
        if kw.get('with_count', False):
            return (200, {}, {'backups': [_stub_backup_full(backup1, base_uri, tenant_id), _stub_backup_full(backup2, base_uri, tenant_id)], 'count': 2})
        return (200, {}, {'backups': [_stub_backup_full(backup1, base_uri, tenant_id), _stub_backup_full(backup2, base_uri, tenant_id)]})

    def delete_backups_76a17945_3c6f_435c_975b_b5685db10b62(self, **kw):
        return (202, {}, None)

    def delete_backups_1234(self, **kw):
        return (202, {}, None)

    def delete_backups_5678(self, **kw):
        return (202, {}, None)

    def post_backups(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        backup1 = '76a17945-3c6f-435c-975b-b5685db10b62'
        return (202, {}, {'backup': _stub_backup(backup1, base_uri, tenant_id)})

    def post_backups_76a17945_3c6f_435c_975b_b5685db10b62_restore(self, **kw):
        return (200, {}, {'restore': _stub_restore()})

    def post_backups_1234_restore(self, **kw):
        return (200, {}, {'restore': _stub_restore()})

    def post_backups_76a17945_3c6f_435c_975b_b5685db10b62_action(self, **kw):
        return (200, {}, None)

    def post_backups_1234_action(self, **kw):
        return (200, {}, None)

    def post_backups_5678_action(self, **kw):
        return (200, {}, None)

    def get_backups_76a17945_3c6f_435c_975b_b5685db10b62_export_record(self, **kw):
        return (200, {}, {'backup-record': {'backup_service': 'fake-backup-service', 'backup_url': 'fake-backup-url'}})

    def get_backups_1234_export_record(self, **kw):
        return (200, {}, {'backup-record': {'backup_service': 'fake-backup-service', 'backup_url': 'fake-backup-url'}})

    def post_backups_import_record(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        backup1 = '76a17945-3c6f-435c-975b-b5685db10b62'
        return (200, {}, {'backup': _stub_backup(backup1, base_uri, tenant_id)})

    def get_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        qos_id1 = '1B6B6A04-A927-4AEB-810B-B7BAAD49F57C'
        return (200, {}, _stub_qos_full(qos_id1, base_uri, tenant_id))

    def get_qos_specs(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        qos_id1 = '1B6B6A04-A927-4AEB-810B-B7BAAD49F57C'
        qos_id2 = '0FD8DD14-A396-4E55-9573-1FE59042E95B'
        return (200, {}, {'qos_specs': [_stub_qos_full(qos_id1, base_uri, tenant_id, 'name-1'), _stub_qos_full(qos_id2, base_uri, tenant_id)]})

    def post_qos_specs(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        qos_id = '1B6B6A04-A927-4AEB-810B-B7BAAD49F57C'
        qos_name = 'qos-name'
        return (202, {}, _stub_qos_full(qos_id, base_uri, tenant_id, qos_name))

    def put_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C(self, **kw):
        return (202, {}, None)

    def put_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C_delete_keys(self, **kw):
        return (202, {}, None)

    def delete_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C(self, **kw):
        return (202, {}, None)

    def get_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C_associations(self, **kw):
        type_id1 = '4230B13A-7A37-4E84-B777-EFBA6FCEE4FF'
        type_id2 = '4230B13A-AB37-4E84-B777-EFBA6FCEE4FF'
        type_name1 = 'type1'
        type_name2 = 'type2'
        return (202, {}, {'qos_associations': [_stub_qos_associates(type_id1, type_name1), _stub_qos_associates(type_id2, type_name2)]})

    def get_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C_associate(self, **kw):
        return (202, {}, None)

    def get_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C_disassociate(self, **kw):
        return (202, {}, None)

    def get_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C_disassociate_all(self, **kw):
        return (202, {}, None)

    def get_os_volume_transfer_5678(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        transfer1 = '5678'
        return (200, {}, {'transfer': _stub_transfer_full(transfer1, base_uri, tenant_id)})

    def get_os_volume_transfer_detail(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        transfer1 = '5678'
        transfer2 = 'f625ec3e-13dd-4498-a22a-50afd534cc41'
        return (200, {}, {'transfers': [_stub_transfer_full(transfer1, base_uri, tenant_id), _stub_transfer_full(transfer2, base_uri, tenant_id)]})

    def delete_os_volume_transfer_5678(self, **kw):
        return (202, {}, None)

    def post_os_volume_transfer(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        transfer1 = '5678'
        return (202, {}, {'transfer': _stub_transfer(transfer1, base_uri, tenant_id)})

    def post_os_volume_transfer_5678_accept(self, **kw):
        base_uri = 'http://localhost:8776'
        tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
        transfer1 = '5678'
        return (200, {}, {'transfer': _stub_transfer(transfer1, base_uri, tenant_id)})

    def get_with_base_url(self, url, **kw):
        if 'default-types' in url:
            return self._cs_request(url, 'GET', **kw)
        server_versions = _stub_server_versions()
        return (200, {'versions': server_versions})

    def create_update_with_base_url(self, url, **kwargs):
        return self._cs_request(url, 'PUT', **kwargs)

    def put_v3_default_types_629632e7_99d2_4c40_9ae3_106fa3b1c9b7(self, **kwargs):
        default_type = stub_default_type()
        return (200, {}, default_type)

    def get_v3_default_types_629632e7_99d2_4c40_9ae3_106fa3b1c9b7(self, **kw):
        default_types = stub_default_type()
        return (200, {}, default_types)

    def get_v3_default_types(self, **kw):
        default_types = stub_default_types()
        return (200, {}, default_types)

    def delete_with_base_url(self, url, **kwargs):
        return self._cs_request(url, 'DELETE', **kwargs)

    def delete_v3_default_types_629632e7_99d2_4c40_9ae3_106fa3b1c9b7(self, **kwargs):
        return (204, {}, {})

    def get_os_services(self, **kw):
        host = kw.get('host', None)
        binary = kw.get('binary', None)
        services = [{'binary': 'cinder-volume', 'host': 'host1', 'zone': 'cinder', 'status': 'enabled', 'state': 'up', 'updated_at': datetime(2012, 10, 29, 13, 42, 2)}, {'binary': 'cinder-volume', 'host': 'host2', 'zone': 'cinder', 'status': 'disabled', 'state': 'down', 'updated_at': datetime(2012, 9, 18, 8, 3, 38)}, {'binary': 'cinder-scheduler', 'host': 'host2', 'zone': 'cinder', 'status': 'disabled', 'state': 'down', 'updated_at': datetime(2012, 9, 18, 8, 3, 38)}]
        if host:
            services = [i for i in services if i['host'] == host]
        if binary:
            services = [i for i in services if i['binary'] == binary]
        return (200, {}, {'services': services})

    def put_os_services_enable(self, body, **kw):
        return (200, {}, {'host': body['host'], 'binary': body['binary'], 'status': 'enabled'})

    def put_os_services_disable(self, body, **kw):
        return (200, {}, {'host': body['host'], 'binary': body['binary'], 'status': 'disabled'})

    def put_os_services_disable_log_reason(self, body, **kw):
        return (200, {}, {'host': body['host'], 'binary': body['binary'], 'status': 'disabled', 'disabled_reason': body['disabled_reason']})

    def get_os_availability_zone(self, **kw):
        return (200, {}, {'availabilityZoneInfo': [{'zoneName': 'zone-1', 'zoneState': {'available': True}, 'hosts': None}, {'zoneName': 'zone-2', 'zoneState': {'available': False}, 'hosts': None}]})

    def get_os_availability_zone_detail(self, **kw):
        return (200, {}, {'availabilityZoneInfo': [{'zoneName': 'zone-1', 'zoneState': {'available': True}, 'hosts': {'fake_host-1': {'cinder-volume': {'active': True, 'available': True, 'updated_at': datetime(2012, 12, 26, 14, 45, 25, 0)}}}}, {'zoneName': 'internal', 'zoneState': {'available': True}, 'hosts': {'fake_host-1': {'cinder-sched': {'active': True, 'available': True, 'updated_at': datetime(2012, 12, 26, 14, 45, 24, 0)}}}}, {'zoneName': 'zone-2', 'zoneState': {'available': False}, 'hosts': None}]})

    def post_snapshots_1234_metadata(self, **kw):
        return (200, {}, {'metadata': {'key1': 'val1', 'key2': 'val2'}})

    def delete_snapshots_1234_metadata_key1(self, **kw):
        return (200, {}, None)

    def delete_snapshots_1234_metadata_key2(self, **kw):
        return (200, {}, None)

    def put_volumes_1234_metadata(self, **kw):
        return (200, {}, {'metadata': {'key1': 'val1', 'key2': 'val2'}})

    def put_snapshots_1234_metadata(self, **kw):
        return (200, {}, {'metadata': {'key1': 'val1', 'key2': 'val2'}})

    def get_os_volume_manage(self, **kw):
        vol_id = 'volume-ffffffff-0000-ffff-0000-ffffffffffff'
        vols = [{'size': 4, 'safe_to_manage': False, 'actual_size': 4.0, 'reference': {'source-name': vol_id}}, {'size': 5, 'safe_to_manage': True, 'actual_size': 4.3, 'reference': {'source-name': 'myvol'}}]
        return (200, {}, {'manageable-volumes': vols})

    def get_os_volume_manage_detail(self, **kw):
        vol_id = 'volume-ffffffff-0000-ffff-0000-ffffffffffff'
        vols = [{'size': 4, 'reason_not_safe': 'volume in use', 'safe_to_manage': False, 'extra_info': 'qos_setting:high', 'reference': {'source-name': vol_id}, 'actual_size': 4.0}, {'size': 5, 'reason_not_safe': None, 'safe_to_manage': True, 'extra_info': 'qos_setting:low', 'actual_size': 4.3, 'reference': {'source-name': 'myvol'}}]
        return (200, {}, {'manageable-volumes': vols})

    def post_os_volume_manage(self, **kw):
        volume = _stub_volume(id='1234')
        volume.update(kw['body']['volume'])
        return (202, {}, {'volume': volume})

    def get_os_snapshot_manage(self, **kw):
        snap_id = 'snapshot-ffffffff-0000-ffff-0000-ffffffffffff'
        snaps = [{'actual_size': 4.0, 'size': 4, 'safe_to_manage': False, 'source_id_type': 'source-name', 'source_cinder_id': '00000000-ffff-0000-ffff-00000000', 'reference': {'source-name': snap_id}, 'source_identifier': 'volume-00000000-ffff-0000-ffff-000000'}, {'actual_size': 4.3, 'reference': {'source-name': 'mysnap'}, 'source_id_type': 'source-name', 'source_identifier': 'myvol', 'safe_to_manage': True, 'source_cinder_id': None, 'size': 5}]
        return (200, {}, {'manageable-snapshots': snaps})

    def get_os_snapshot_manage_detail(self, **kw):
        snap_id = 'snapshot-ffffffff-0000-ffff-0000-ffffffffffff'
        snaps = [{'actual_size': 4.0, 'size': 4, 'safe_to_manage': False, 'source_id_type': 'source-name', 'source_cinder_id': '00000000-ffff-0000-ffff-00000000', 'reference': {'source-name': snap_id}, 'source_identifier': 'volume-00000000-ffff-0000-ffff-000000', 'extra_info': 'qos_setting:high', 'reason_not_safe': 'snapshot in use'}, {'actual_size': 4.3, 'reference': {'source-name': 'mysnap'}, 'safe_to_manage': True, 'source_cinder_id': None, 'source_id_type': 'source-name', 'identifier': 'mysnap', 'source_identifier': 'myvol', 'size': 5, 'extra_info': 'qos_setting:low', 'reason_not_safe': None}]
        return (200, {}, {'manageable-snapshots': snaps})

    def post_os_snapshot_manage(self, **kw):
        snapshot = _stub_snapshot(id='1234', volume_id='volume_id1')
        snapshot.update(kw['body']['snapshot'])
        return (202, {}, {'snapshot': snapshot})

    def get_scheduler_stats_get_pools(self, **kw):
        stats = [{'name': 'ubuntu@lvm#backend_name', 'capabilities': {'pool_name': 'backend_name', 'QoS_support': False, 'timestamp': '2014-11-21T18:15:28.141161', 'allocated_capacity_gb': 0, 'volume_backend_name': 'backend_name', 'free_capacity_gb': 7.01, 'driver_version': '2.0.0', 'total_capacity_gb': 10.01, 'reserved_percentage': 0, 'vendor_name': 'Open Source', 'storage_protocol': 'iSCSI'}}]
        return (200, {}, {'pools': stats})

    def get_capabilities_host(self, **kw):
        return (200, {}, {'namespace': 'OS::Storage::Capabilities::fake', 'vendor_name': 'OpenStack', 'volume_backend_name': 'lvm', 'pool_name': 'pool', 'storage_protocol': 'iSCSI', 'properties': {'compression': {'title': 'Compression', 'description': 'Enables compression.', 'type': 'boolean'}}})