class TestHypothesisStrategies(HyperlinkTestCase):
    """
        Tests for hyperlink.hypothesis.
        """

    @given(idna_text())
    def test_idna_text_valid(self, text):
        """
            idna_text() generates IDNA-encodable text.
            """
        try:
            idna_encode(text)
        except IDNAError:
            raise AssertionError('Invalid IDNA text: {!r}'.format(text))

    @given(data())
    def test_idna_text_min_max(self, data):
        """
            idna_text() raises AssertionError if min_size is < 1.
            """
        self.assertRaises(AssertionError, data.draw, idna_text(min_size=0))
        self.assertRaises(AssertionError, data.draw, idna_text(max_size=0))

    @given(port_numbers())
    def test_port_numbers_bounds(self, port):
        """
            port_numbers() generates integers between 1 and 65535, inclusive.
            """
        self.assertGreaterEqual(port, 1)
        self.assertLessEqual(port, 65535)

    @given(port_numbers(allow_zero=True))
    def test_port_numbers_bounds_allow_zero(self, port):
        """
            port_numbers(allow_zero=True) generates integers between 0 and
            65535, inclusive.
            """
        self.assertGreaterEqual(port, 0)
        self.assertLessEqual(port, 65535)

    @given(hostname_labels())
    def test_hostname_labels_valid_idn(self, label):
        """
            hostname_labels() generates IDN host name labels.
            """
        try:
            check_label(label)
            idna_encode(label)
        except UnicodeError:
            raise AssertionError('Invalid IDN label: {!r}'.format(label))

    @given(data())
    @settings(max_examples=10)
    def test_hostname_labels_long_idn_punycode(self, data):
        """
            hostname_labels() handles case where idna_text() generates text
            that encoded to punycode ends up as longer than allowed.
            """

        @composite
        def mock_idna_text(draw, min_size, max_size):
            return u'á' * max_size
        with patch('hyperlink.hypothesis.idna_text', mock_idna_text):
            label = data.draw(hostname_labels())
            try:
                check_label(label)
                idna_encode(label)
            except UnicodeError:
                raise AssertionError('Invalid IDN label: {!r}'.format(label))

    @given(hostname_labels(allow_idn=False))
    def test_hostname_labels_valid_ascii(self, label):
        """
            hostname_labels() generates a ASCII host name labels.
            """
        try:
            check_label(label)
            label.encode('ascii')
        except UnicodeError:
            raise AssertionError('Invalid ASCII label: {!r}'.format(label))

    @given(hostnames())
    def test_hostnames_idn(self, hostname):
        """
            hostnames() generates a IDN host names.
            """
        try:
            for label in hostname.split(u'.'):
                check_label(label)
            idna_encode(hostname)
        except UnicodeError:
            raise AssertionError('Invalid IDN host name: {!r}'.format(hostname))

    @given(hostnames(allow_leading_digit=False))
    def test_hostnames_idn_nolead(self, hostname):
        """
            hostnames(allow_leading_digit=False) generates a IDN host names
            without leading digits.
            """
        self.assertTrue(hostname == hostname.lstrip(digits))

    @given(hostnames(allow_idn=False))
    def test_hostnames_ascii(self, hostname):
        """
            hostnames() generates a ASCII host names.
            """
        try:
            for label in hostname.split(u'.'):
                check_label(label)
            hostname.encode('ascii')
        except UnicodeError:
            raise AssertionError('Invalid ASCII host name: {!r}'.format(hostname))

    @given(hostnames(allow_leading_digit=False, allow_idn=False))
    def test_hostnames_ascii_nolead(self, hostname):
        """
            hostnames(allow_leading_digit=False, allow_idn=False) generates
            ASCII host names without leading digits.
            """
        self.assertTrue(hostname == hostname.lstrip(digits))

    @given(paths())
    def test_paths(self, path):
        """
            paths() generates sequences of URL path components.
            """
        text = u'/'.join(path)
        try:
            text.encode('utf-8')
        except UnicodeError:
            raise AssertionError('Invalid URL path: {!r}'.format(path))
        for segment in path:
            self.assertNotIn('#/?', segment)

    @given(encoded_urls())
    def test_encoded_urls(self, url):
        """
            encoded_urls() generates EncodedURLs.
            """
        self.assertIsInstance(url, EncodedURL)

    @given(decoded_urls())
    def test_decoded_urls(self, url):
        """
            decoded_urls() generates DecodedURLs.
            """
        self.assertIsInstance(url, DecodedURL)