from twisted.python import reflect
class Homogenous(Constrained):
    """A homogenous collection of entities.

    I will only contain entities that are an instance of the class or type
    specified by my 'entityType' attribute.
    """
    entityType = object

    def entityConstraint(self, entity):
        if isinstance(entity, self.entityType):
            return 1
        else:
            raise ConstraintViolation(f'{entity} of incorrect type ({self.entityType})')

    def getNameType(self):
        return 'Name'

    def getEntityType(self):
        return self.entityType.__name__