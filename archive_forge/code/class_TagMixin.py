from openstack import exceptions
from openstack import resource
from openstack import utils
class TagMixin:
    id: resource.Body
    base_path: str
    _body: resource._ComponentManager

    @classmethod
    def _get_session(cls, session):
        ...
    _tag_query_parameters = {'tags': 'tags', 'any_tags': 'tags-any', 'not_tags': 'not-tags', 'not_any_tags': 'not-tags-any'}
    tags = resource.Body('tags', type=list, default=[])

    def fetch_tags(self, session):
        """Lists tags set on the entity.

        :param session: The session to use for making this request.
        :return: The list with tags attached to the entity
        """
        url = utils.urljoin(self.base_path, self.id, 'tags')
        session = self._get_session(session)
        response = session.get(url)
        exceptions.raise_from_response(response)
        json = response.json()
        if 'tags' in json:
            self._body.attributes.update({'tags': json['tags']})
        return self

    def set_tags(self, session, tags=[]):
        """Sets/Replaces all tags on the resource.

        :param session: The session to use for making this request.
        :param list tags: List with tags to be set on the resource
        """
        url = utils.urljoin(self.base_path, self.id, 'tags')
        session = self._get_session(session)
        response = session.put(url, json={'tags': tags})
        exceptions.raise_from_response(response)
        self._body.attributes.update({'tags': tags})
        return self

    def remove_all_tags(self, session):
        """Removes all tags on the entity.

        :param session: The session to use for making this request.
        """
        url = utils.urljoin(self.base_path, self.id, 'tags')
        session = self._get_session(session)
        response = session.delete(url)
        exceptions.raise_from_response(response)
        self._body.attributes.update({'tags': []})
        return self

    def check_tag(self, session, tag):
        """Checks if tag exists on the entity.

        If the tag does not exist a 404 will be returned

        :param session: The session to use for making this request.
        :param tag: The tag as a string.
        """
        url = utils.urljoin(self.base_path, self.id, 'tags', tag)
        session = self._get_session(session)
        response = session.get(url)
        exceptions.raise_from_response(response, error_message='Tag does not exist')
        return self

    def add_tag(self, session, tag):
        """Adds a single tag to the resource.

        :param session: The session to use for making this request.
        :param tag: The tag as a string.
        """
        url = utils.urljoin(self.base_path, self.id, 'tags', tag)
        session = self._get_session(session)
        response = session.put(url)
        exceptions.raise_from_response(response)
        tags = self.tags
        tags.append(tag)
        self._body.attributes.update({'tags': tags})
        return self

    def remove_tag(self, session, tag):
        """Removes a single tag from the specified resource.

        :param session: The session to use for making this request.
        :param tag: The tag as a string.
        """
        url = utils.urljoin(self.base_path, self.id, 'tags', tag)
        session = self._get_session(session)
        response = session.delete(url)
        exceptions.raise_from_response(response)
        tags = self.tags
        try:
            tags.remove(tag)
        except ValueError:
            pass
        self._body.attributes.update({'tags': tags})
        return self