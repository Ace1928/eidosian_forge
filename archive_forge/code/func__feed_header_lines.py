import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def _feed_header_lines(self, consumer, lines):
    consumer_dict = {'DEFINITION': 'definition', 'ACCESSION': 'accession', 'NID': 'nid', 'PID': 'pid', 'DBSOURCE': 'db_source', 'KEYWORDS': 'keywords', 'SEGMENT': 'segment', 'SOURCE': 'source', 'AUTHORS': 'authors', 'CONSRTM': 'consrtm', 'PROJECT': 'project', 'TITLE': 'title', 'JOURNAL': 'journal', 'MEDLINE': 'medline_id', 'PUBMED': 'pubmed_id', 'REMARK': 'remark'}
    lines = [_f for _f in lines if _f]
    lines.append('')
    line_iter = iter(lines)
    try:
        line = next(line_iter)
        while True:
            if not line:
                break
            line_type = line[:self.GENBANK_INDENT].strip()
            data = line[self.GENBANK_INDENT:].strip()
            if line_type == 'VERSION':
                while '  ' in data:
                    data = data.replace('  ', ' ')
                if ' GI:' not in data:
                    consumer.version(data)
                else:
                    if self.debug:
                        print('Version [' + data.split(' GI:')[0] + '], gi [' + data.split(' GI:')[1] + ']')
                    consumer.version(data.split(' GI:')[0])
                    consumer.gi(data.split(' GI:')[1])
                line = next(line_iter)
            elif line_type == 'DBLINK':
                consumer.dblink(data.strip())
                while True:
                    line = next(line_iter)
                    if line[:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                        consumer.dblink(line[self.GENBANK_INDENT:].strip())
                    else:
                        break
            elif line_type == 'REFERENCE':
                if self.debug > 1:
                    print('Found reference [' + data + ']')
                data = data.strip()
                while True:
                    line = next(line_iter)
                    if line[:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                        data += ' ' + line[self.GENBANK_INDENT:]
                        if self.debug > 1:
                            print('Extended reference text [' + data + ']')
                    else:
                        break
                while '  ' in data:
                    data = data.replace('  ', ' ')
                if ' ' not in data:
                    if self.debug > 2:
                        print('Reference number "' + data + '"')
                    consumer.reference_num(data)
                else:
                    if self.debug > 2:
                        print('Reference number "' + data[:data.find(' ')] + '", "' + data[data.find(' ') + 1:] + '"')
                    consumer.reference_num(data[:data.find(' ')])
                    consumer.reference_bases(data[data.find(' ') + 1:])
            elif line_type == 'ORGANISM':
                organism_data = data
                lineage_data = ''
                while True:
                    line = next(line_iter)
                    if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                        if lineage_data or ';' in line or line[self.GENBANK_INDENT:].strip() in ('Bacteria.', 'Archaea.', 'Eukaryota.', 'Unclassified.', 'Viruses.', 'cellular organisms.', 'other sequences.', 'unclassified sequences.'):
                            lineage_data += ' ' + line[self.GENBANK_INDENT:]
                        elif line[self.GENBANK_INDENT:].strip() == '.':
                            pass
                        else:
                            organism_data += ' ' + line[self.GENBANK_INDENT:].strip()
                    else:
                        break
                consumer.organism(organism_data)
                if lineage_data.strip() == '' and self.debug > 1:
                    print('Taxonomy line(s) missing or blank')
                consumer.taxonomy(lineage_data.strip())
                del organism_data, lineage_data
            elif line_type == 'COMMENT':
                data = line[self.GENBANK_INDENT:]
                if self.debug > 1:
                    print('Found comment')
                comment_list = []
                structured_comment_dict = defaultdict(dict)
                regex = f'([^#]+){self.STRUCTURED_COMMENT_START}$'
                structured_comment_key = re.search(regex, data)
                if structured_comment_key is not None:
                    structured_comment_key = structured_comment_key.group(1)
                    if self.debug > 1:
                        print('Found Structured Comment')
                else:
                    comment_list.append(data)
                while True:
                    line = next(line_iter)
                    data = line[self.GENBANK_INDENT:]
                    if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                        if self.STRUCTURED_COMMENT_START in data:
                            regex = f'([^#]+){self.STRUCTURED_COMMENT_START}$'
                            structured_comment_key = re.search(regex, data)
                            if structured_comment_key is not None:
                                structured_comment_key = structured_comment_key.group(1)
                            else:
                                comment_list.append(data)
                        elif structured_comment_key is not None and self.STRUCTURED_COMMENT_DELIM.strip() in data:
                            match = re.search(f'(.+?)\\s*{self.STRUCTURED_COMMENT_DELIM.strip()}\\s*(.*)', data)
                            structured_comment_dict[structured_comment_key][match.group(1)] = match.group(2)
                            if self.debug > 2:
                                print('Structured Comment continuation [' + data + ']')
                        elif structured_comment_key is not None and self.STRUCTURED_COMMENT_END not in data:
                            if structured_comment_key not in structured_comment_dict:
                                warnings.warn('Structured comment not parsed for %s. Is it malformed?' % consumer.data.name, BiopythonParserWarning)
                                continue
                            previous_value_line = structured_comment_dict[structured_comment_key][match.group(1)]
                            structured_comment_dict[structured_comment_key][match.group(1)] = previous_value_line + ' ' + line.strip()
                        elif self.STRUCTURED_COMMENT_END in data:
                            structured_comment_key = None
                        else:
                            comment_list.append(data)
                            if self.debug > 2:
                                print('Comment continuation [' + data + ']')
                    else:
                        break
                if comment_list:
                    consumer.comment(comment_list)
                if structured_comment_dict:
                    consumer.structured_comment(structured_comment_dict)
                del comment_list, structured_comment_key, structured_comment_dict
            elif line_type in consumer_dict:
                while True:
                    line = next(line_iter)
                    if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                        data += ' ' + line[self.GENBANK_INDENT:]
                    else:
                        if line_type == 'DEFINITION' and data.endswith('.'):
                            data = data[:-1]
                        getattr(consumer, consumer_dict[line_type])(data)
                        break
            else:
                if self.debug:
                    print('Ignoring GenBank header line:\n' % line)
                line = next(line_iter)
    except StopIteration:
        raise ValueError('Problem in header') from None