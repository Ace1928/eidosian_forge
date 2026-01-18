from django.db.backends.oracle.introspection import DatabaseIntrospection
from django.db.backends.oracle.oracledb_any import oracledb
from django.utils.functional import cached_property
class OracleIntrospection(DatabaseIntrospection):

    @cached_property
    def data_types_reverse(self):
        return {**super().data_types_reverse, oracledb.DB_TYPE_OBJECT: 'GeometryField'}

    def get_geometry_type(self, table_name, description):
        with self.connection.cursor() as cursor:
            try:
                cursor.execute('SELECT "DIMINFO", "SRID" FROM "USER_SDO_GEOM_METADATA" WHERE "TABLE_NAME"=%s AND "COLUMN_NAME"=%s', (table_name.upper(), description.name.upper()))
                row = cursor.fetchone()
            except Exception as exc:
                raise Exception('Could not find entry in USER_SDO_GEOM_METADATA corresponding to "%s"."%s"' % (table_name, description.name)) from exc
            field_type = 'GeometryField'
            field_params = {}
            dim, srid = row
            if srid != 4326:
                field_params['srid'] = srid
            dim = dim.size()
            if dim != 2:
                field_params['dim'] = dim
        return (field_type, field_params)