from information_schema.collations order by id;" | python -c "import sys
class Charsets:

    def __init__(self):
        self._by_id = {}
        self._by_name = {}

    def add(self, c):
        self._by_id[c.id] = c
        if c.is_default:
            self._by_name[c.name] = c

    def by_id(self, id):
        return self._by_id[id]

    def by_name(self, name):
        if name == 'utf8':
            name = 'utf8mb4'
        return self._by_name.get(name.lower())