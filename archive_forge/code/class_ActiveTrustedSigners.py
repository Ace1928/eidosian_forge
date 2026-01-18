class ActiveTrustedSigners(list):

    def startElement(self, name, attrs, connection):
        if name == 'Signer':
            s = Signer()
            self.append(s)
            return s

    def endElement(self, name, value, connection):
        pass