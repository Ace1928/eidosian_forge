from pecan import expose, response, abort
class PeopleController(object):

    @expose()
    def _lookup(self, person_id, *remainder):
        return (PersonController(int(person_id)), remainder)

    @expose(generic=True, template='json')
    def index(self):
        return people

    @index.when(method='POST', template='json')
    def post(self):
        response.status = 201