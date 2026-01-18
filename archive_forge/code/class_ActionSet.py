from boto.ec2.instancestatus import Status, Details
class ActionSet(list):

    def startElement(self, name, attrs, connection):
        if name == 'item':
            action = Action()
            self.append(action)
            return action
        else:
            return None

    def endElement(self, name, value, connection):
        setattr(self, name, value)