class ISchema(IValidator):
    fields = Attribute('A dictionary of (field name: validator)', name='fields')