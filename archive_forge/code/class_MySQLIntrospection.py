from MySQLdb.constants import FIELD_TYPE
from django.contrib.gis.gdal import OGRGeomType
from django.db.backends.mysql.introspection import DatabaseIntrospection
class MySQLIntrospection(DatabaseIntrospection):
    data_types_reverse = DatabaseIntrospection.data_types_reverse.copy()
    data_types_reverse[FIELD_TYPE.GEOMETRY] = 'GeometryField'

    def get_geometry_type(self, table_name, description):
        with self.connection.cursor() as cursor:
            cursor.execute('DESCRIBE %s' % self.connection.ops.quote_name(table_name))
            for column, typ, null, key, default, extra in cursor.fetchall():
                if column == description.name:
                    field_type = OGRGeomType(typ).django
                    field_params = {}
                    break
        return (field_type, field_params)

    def supports_spatial_index(self, cursor, table_name):
        storage_engine = self.get_storage_engine(cursor, table_name)
        return storage_engine in ('MyISAM', 'Aria', 'InnoDB')