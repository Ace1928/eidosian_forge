from dissononce.dh.public import PublicKey

        Performs a Diffie-Hellman calculation between the private key in key_pair and the public_key and returns an
        output sequence of bytes of length DHLEN

        :param key_pair:
        :type key_pair: dissononce.dh.keypair.KeyPair
        :param public_key:
        :type public_key: dissononce.dh.public.PublicKey
        :return:
        :rtype: bytes
        