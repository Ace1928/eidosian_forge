from wsme.utils import _
def add_fieldname(self, name):
    """Add a fieldname to concatenate the full name.

        Add a fieldname so that the whole hierarchy is displayed. Successive
        calls to this method will prepend ``name`` to the hierarchy of names.
        """
    if self.fieldname is not None:
        self.fieldname = '{}.{}'.format(name, self.fieldname)
    else:
        self.fieldname = name
    super(UnknownAttribute, self).__init__(self.msg)