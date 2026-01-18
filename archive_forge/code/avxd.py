from gensim.summarization import summarize

# News articles
articles = [
    "The government announced a new economic stimulus package to boost the country's economy. The package includes tax cuts, infrastructure investments, and support for small businesses. Experts believe that the measures will help create jobs and stimulate economic growth. However, some critics argue that the package may not be sufficient to address the long-term challenges faced by the economy.",
    "A major technology company unveiled its latest smartphone model at a highly anticipated event. The new device features a larger screen, improved camera capabilities, and enhanced performance. The company claims that the phone will revolutionize the smartphone industry and set new standards for innovation. Pre-orders for the device have already begun, and it is expected to hit the market next month.",
]

# Generate a summary for each article
for article in articles:
    summary = summarize(article, ratio=0.3)

    print("Article:", article)
    print("Summary:", summary)
    print("---")
