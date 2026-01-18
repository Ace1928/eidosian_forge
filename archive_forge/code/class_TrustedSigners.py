class TrustedSigners(list):

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Self':
            self.append(name)
        elif name == 'AwsAccountNumber':
            self.append(value)