import io
import json
import mimetypes
from sentry_sdk._compat import text_type, PY2
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.session import Session
from sentry_sdk.utils import json_dumps, capture_internal_exceptions
class Envelope(object):

    def __init__(self, headers=None, items=None):
        if headers is not None:
            headers = dict(headers)
        self.headers = headers or {}
        if items is None:
            items = []
        else:
            items = list(items)
        self.items = items

    @property
    def description(self):
        return 'envelope with %s items (%s)' % (len(self.items), ', '.join((x.data_category for x in self.items)))

    def add_event(self, event):
        self.add_item(Item(payload=PayloadRef(json=event), type='event'))

    def add_transaction(self, transaction):
        self.add_item(Item(payload=PayloadRef(json=transaction), type='transaction'))

    def add_profile(self, profile):
        self.add_item(Item(payload=PayloadRef(json=profile), type='profile'))

    def add_checkin(self, checkin):
        self.add_item(Item(payload=PayloadRef(json=checkin), type='check_in'))

    def add_session(self, session):
        if isinstance(session, Session):
            session = session.to_json()
        self.add_item(Item(payload=PayloadRef(json=session), type='session'))

    def add_sessions(self, sessions):
        self.add_item(Item(payload=PayloadRef(json=sessions), type='sessions'))

    def add_item(self, item):
        self.items.append(item)

    def get_event(self):
        for items in self.items:
            event = items.get_event()
            if event is not None:
                return event
        return None

    def get_transaction_event(self):
        for item in self.items:
            event = item.get_transaction_event()
            if event is not None:
                return event
        return None

    def __iter__(self):
        return iter(self.items)

    def serialize_into(self, f):
        f.write(json_dumps(self.headers))
        f.write(b'\n')
        for item in self.items:
            item.serialize_into(f)

    def serialize(self):
        out = io.BytesIO()
        self.serialize_into(out)
        return out.getvalue()

    @classmethod
    def deserialize_from(cls, f):
        headers = parse_json(f.readline())
        items = []
        while 1:
            item = Item.deserialize_from(f)
            if item is None:
                break
            items.append(item)
        return cls(headers=headers, items=items)

    @classmethod
    def deserialize(cls, bytes):
        return cls.deserialize_from(io.BytesIO(bytes))

    def __repr__(self):
        return '<Envelope headers=%r items=%r>' % (self.headers, self.items)