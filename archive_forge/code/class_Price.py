class Price(object):

    def __init__(self, amount=0.0, currency_code='USD'):
        self.amount = amount
        self.currency_code = currency_code
        self.formatted_price = ''

    def __repr__(self):
        if self.formatted_price:
            return self.formatted_price
        else:
            return str(self.amount)

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Amount':
            self.amount = float(value)
        elif name == 'CurrencyCode':
            self.currency_code = value
        elif name == 'FormattedPrice':
            self.formatted_price = value

    def get_as_params(self, label, ord=1):
        return {'%s.%d.Amount' % (label, ord): str(self.amount), '%s.%d.CurrencyCode' % (label, ord): self.currency_code}