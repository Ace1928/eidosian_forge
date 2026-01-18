from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# News article
article = """
The government announced a new economic stimulus package to boost the country's economy. The package includes tax cuts, infrastructure investments, and support for small businesses. Experts believe that the measures will help create jobs and stimulate economic growth. However, some critics argue that the package may not be sufficient to address the long-term challenges faced by the economy.

The stimulus package comes amid concerns over the slowdown in economic activity due to the ongoing pandemic. The government hopes that the measures will provide a much-needed boost to consumer spending and business confidence. The tax cuts are expected to put more money in the hands of individuals and businesses, while the infrastructure investments aim to create jobs and improve the country's competitiveness.

Small businesses, which have been hit hard by the pandemic, will receive additional support through grants and loans. The government recognizes the importance of small businesses in driving economic growth and employment. The package also includes measures to support the tourism and hospitality sectors, which have been severely impacted by travel restrictions.

Critics of the stimulus package argue that it may not be enough to address the structural issues facing the economy. They point out that the country's high debt levels and declining productivity growth require long-term solutions. Some economists also warn that the stimulus measures could lead to inflationary pressures if not managed carefully.

Despite the concerns, the government remains optimistic about the impact of the stimulus package. They believe that the measures will provide a significant boost to the economy and help the country recover from the pandemic-induced recession. The government has also pledged to continue monitoring the economic situation and take further action if necessary.
"""

# Create a plaintext parser
parser = PlaintextParser.from_string(article, Tokenizer("english"))

# Create a LexRank summarizer
summarizer = LexRankSummarizer()

# Generate the summary
summary = summarizer(parser.document, sentences_count=3)

# Print the summary
for sentence in summary:
    print(sentence)
