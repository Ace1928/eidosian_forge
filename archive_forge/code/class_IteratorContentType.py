from inspect import isclass
class IteratorContentType(DependentType):
    """
            Type of an iterator over the content of a container
            """

    def generate(self, ctx):
        iterator_value_type = ctx(self.of)
        return 'typename std::remove_cv<{0}>::type'.format('typename std::iterator_traits<{0}>::value_type'.format('typename std::remove_reference<{0}>::type::iterator'.format(iterator_value_type)))