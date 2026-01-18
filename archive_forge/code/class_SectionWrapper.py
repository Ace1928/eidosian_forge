class SectionWrapper(object):

    def __init__(self, config, name):
        self.config = config
        self.name = name

    def lineof(self, name):
        return self.config.lineof(self.name, name)

    def get(self, key, default=None, convert=str):
        return self.config.get(self.name, key, convert=convert, default=default)

    def __getitem__(self, key):
        return self.config.sections[self.name][key]

    def __iter__(self):
        section = self.config.sections.get(self.name, [])

        def lineof(key):
            return self.config.lineof(self.name, key)
        for name in sorted(section, key=lineof):
            yield name

    def items(self):
        for name in self:
            yield (name, self[name])