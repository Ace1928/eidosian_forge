from datetime import datetime, date
from decimal import Decimal
from json import JSONEncoder
class GenericJSON(JSONEncoder):
    """
    Generic JSON encoder.  Makes several attempts to correctly JSONify
    requested response objects.
    """

    def default(self, obj):
        """
        Converts an object and returns a ``JSON``-friendly structure.

        :param obj: object or structure to be converted into a
                    ``JSON``-ifiable structure

        Considers the following special cases in order:

        * object has a callable __json__() attribute defined
            returns the result of the call to __json__()
        * date and datetime objects
            returns the object cast to str
        * Decimal objects
            returns the object cast to float
        * SQLAlchemy objects
            returns a copy of the object.__dict__ with internal SQLAlchemy
            parameters removed
        * SQLAlchemy ResultProxy objects
            Casts the iterable ResultProxy into a list of tuples containing
            the entire resultset data, returns the list in a dictionary
            along with the resultset "row" count.

            .. note:: {'count': 5, 'rows': [('Ed Jones',), ('Pete Jones',),
                ('Wendy Williams',), ('Mary Contrary',), ('Fred Smith',)]}

        * SQLAlchemy RowProxy objects
            Casts the RowProxy cursor object into a dictionary, probably
            losing its ordered dictionary behavior in the process but
            making it JSON-friendly.
        * webob_dicts objects
            returns webob_dicts.mixed() dictionary, which is guaranteed
            to be JSON-friendly.
        """
        if hasattr(obj, '__json__') and callable(obj.__json__):
            return obj.__json__()
        elif isinstance(obj, (date, datetime)):
            return str(obj)
        elif isinstance(obj, Decimal):
            return float(obj)
        elif is_saobject(obj):
            props = {}
            for key in obj.__dict__:
                if not key.startswith('_sa_'):
                    props[key] = getattr(obj, key)
            return props
        elif isinstance(obj, ResultProxy):
            props = dict(rows=list(obj), count=obj.rowcount)
            if props['count'] < 0:
                props['count'] = len(props['rows'])
            return props
        elif isinstance(obj, LegacyCursorResult):
            rows = [dict(row._mapping) for row in obj.fetchall()]
            return {'count': len(rows), 'rows': rows}
        elif isinstance(obj, LegacyRow):
            return dict(obj._mapping)
        elif isinstance(obj, RowProxy):
            if obj.__class__.__name__ == 'Row':
                obj = obj._mapping
            return dict(obj)
        elif isinstance(obj, webob_dicts):
            return obj.mixed()
        else:
            return JSONEncoder.default(self, obj)