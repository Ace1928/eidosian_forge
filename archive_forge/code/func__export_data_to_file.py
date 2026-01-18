import json
import os
from os.path import isfile
from os.path import join
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import select
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def _export_data_to_file(meta, conn, path):
    if not path:
        path = CONF.metadata_source_path
    namespace_table = get_metadef_namespaces_table(meta)
    with conn.begin():
        namespaces = conn.execute(namespace_table.select()).fetchall()
    pattern = re.compile('[\\W_]+', re.UNICODE)
    for id, namespace in enumerate(namespaces, start=1):
        namespace_id = namespace['id']
        namespace_file_name = pattern.sub('', namespace['display_name'])
        values = {'namespace': namespace['namespace'], 'display_name': namespace['display_name'], 'description': namespace['description'], 'visibility': namespace['visibility'], 'protected': namespace['protected'], 'resource_type_associations': [], 'properties': {}, 'objects': [], 'tags': []}
        namespace_resource_types = _get_namespace_resource_types(meta, conn, namespace_id)
        db_objects = _get_objects(meta, conn, namespace_id)
        db_properties = _get_properties(meta, conn, namespace_id)
        db_tags = _get_tags(meta, conn, namespace_id)
        resource_types = []
        for namespace_resource_type in namespace_resource_types:
            resource_type = _get_resource_type(meta, conn, namespace_resource_type['resource_type_id'])
            resource_types.append({'name': resource_type['name'], 'prefix': namespace_resource_type['prefix'], 'properties_target': namespace_resource_type['properties_target']})
        values.update({'resource_type_associations': resource_types})
        objects = []
        for object in db_objects:
            objects.append({'name': object['name'], 'description': object['description'], 'properties': json.loads(object['json_schema'])})
        values.update({'objects': objects})
        properties = {}
        for property in db_properties:
            properties.update({property['name']: json.loads(property['json_schema'])})
        values.update({'properties': properties})
        tags = []
        for tag in db_tags:
            tags.append({'name': tag['name']})
        values.update({'tags': tags})
        try:
            file_name = ''.join([path, namespace_file_name, '.json'])
            if isfile(file_name):
                LOG.info(_LI('Overwriting: %s'), file_name)
            with open(file_name, 'w') as json_file:
                json_file.write(json.dumps(values))
        except Exception as e:
            LOG.exception(encodeutils.exception_to_unicode(e))
        LOG.info(_LI('Namespace %(namespace)s saved in %(file)s'), {'namespace': namespace_file_name, 'file': file_name})