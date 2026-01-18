from boto.compat import six
class Transitions(list):
    """
    A container for the transitions associated with a Lifecycle's Rule configuration.
    """

    def __init__(self):
        self.transition_properties = 3
        self.current_transition_property = 1
        self.temp_days = None
        self.temp_date = None
        self.temp_storage_class = None

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Days':
            self.temp_days = int(value)
        elif name == 'Date':
            self.temp_date = value
        elif name == 'StorageClass':
            self.temp_storage_class = value
        if self.current_transition_property == self.transition_properties:
            self.append(Transition(self.temp_days, self.temp_date, self.temp_storage_class))
            self.temp_days = self.temp_date = self.temp_storage_class = None
            self.current_transition_property = 1
        else:
            self.current_transition_property += 1

    def to_xml(self):
        """
        Returns a string containing the XML version of the Lifecycle
        configuration as defined by S3.
        """
        s = ''
        for transition in self:
            s += transition.to_xml()
        return s

    def add_transition(self, days=None, date=None, storage_class=None):
        """
        Add a transition to this Lifecycle configuration.  This only adds
        the rule to the local copy.  To install the new rule(s) on
        the bucket, you need to pass this Lifecycle config object
        to the configure_lifecycle method of the Bucket object.

        :ivar days: The number of days until the object should be moved.

        :ivar date: The date when the object should be moved.  Should be
            in ISO 8601 format.

        :ivar storage_class: The storage class to transition to.  Valid
            values are GLACIER, STANDARD_IA.
        """
        transition = Transition(days, date, storage_class)
        self.append(transition)

    def __first_or_default(self, prop):
        for transition in self:
            return getattr(transition, prop)
        return None

    @property
    def days(self):
        return self.__first_or_default('days')

    @property
    def date(self):
        return self.__first_or_default('date')

    @property
    def storage_class(self):
        return self.__first_or_default('storage_class')