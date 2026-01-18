import time
from boto.compat import json
class OptionStatus(dict):
    """
    Presents a combination of status field (defined below) which are
    accessed as attributes and option values which are stored in the
    native Python dictionary.  In this class, the option values are
    merged from a JSON object that is stored as the Option part of
    the object.

    :ivar domain_name: The name of the domain this option is associated with.
    :ivar create_date: A timestamp for when this option was created.
    :ivar state: The state of processing a change to an option.
        Possible values:

        * RequiresIndexDocuments: the option's latest value will not
          be visible in searches until IndexDocuments has been called
          and indexing is complete.
        * Processing: the option's latest value is not yet visible in
          all searches but is in the process of being activated.
        * Active: the option's latest value is completely visible.

    :ivar update_date: A timestamp for when this option was updated.
    :ivar update_version: A unique integer that indicates when this
        option was last updated.
    """

    def __init__(self, domain, data=None, refresh_fn=None, save_fn=None):
        self.domain = domain
        self.refresh_fn = refresh_fn
        self.save_fn = save_fn
        self.refresh(data)

    def _update_status(self, status):
        self.creation_date = status['creation_date']
        self.status = status['state']
        self.update_date = status['update_date']
        self.update_version = int(status['update_version'])

    def _update_options(self, options):
        if options:
            self.update(json.loads(options))

    def refresh(self, data=None):
        """
        Refresh the local state of the object.  You can either pass
        new state data in as the parameter ``data`` or, if that parameter
        is omitted, the state data will be retrieved from CloudSearch.
        """
        if not data:
            if self.refresh_fn:
                data = self.refresh_fn(self.domain.name)
        if data:
            self._update_status(data['status'])
            self._update_options(data['options'])

    def to_json(self):
        """
        Return the JSON representation of the options as a string.
        """
        return json.dumps(self)

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'CreationDate':
            self.created = value
        elif name == 'State':
            self.state = value
        elif name == 'UpdateDate':
            self.updated = value
        elif name == 'UpdateVersion':
            self.update_version = int(value)
        elif name == 'Options':
            self.update_from_json_doc(value)
        else:
            setattr(self, name, value)

    def save(self):
        """
        Write the current state of the local object back to the
        CloudSearch service.
        """
        if self.save_fn:
            data = self.save_fn(self.domain.name, self.to_json())
            self.refresh(data)

    def wait_for_state(self, state):
        """
        Performs polling of CloudSearch to wait for the ``state``
        of this object to change to the provided state.
        """
        while self.state != state:
            time.sleep(5)
            self.refresh()